# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
CLI entry point for growth training. Equivalent to running the original
``python main_growth_node_new_p.py ...`` script; once this package is
installed, use ``growing-nn-train-growth`` (see ``pyproject.toml``) or
``python -m growing_nn.cli.train_growth``.

NOTE ON BEHAVIOR CHANGES FROM THE ORIGINAL SCRIPT
--------------------------------------------------
This module preserves the original training logic as closely as possible.
Two pre-existing bugs in the source script are fixed here because they
would otherwise make the script crash outright (not "changed behavior" in
any working configuration, since the buggy paths never successfully ran):

1. The original script did ``import models_v2`` at module scope, but
   ``models_v2.py`` does not exist anywhere in the project — this raised
   ``ModuleNotFoundError`` before ``main()`` could even be defined. That
   import (and the now-nonexistent module) is dropped.
2. ``utils.init_wandb`` / ``utils.log_wandb`` were called but never
   defined in ``utils.py`` — seed
   :mod:`growing_nn.utils.wandb_utils` for the fix; ``--wandb`` now
   actually logs to Weights & Biases instead of raising ``AttributeError``.

See ``MIGRATION.md`` at the repository root for the full list of changes.
"""
import argparse
import datetime
import gc
import json
import os
import time
import warnings
from pathlib import Path

import GPUtil
import psutil
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler

from growing_nn import models  # noqa: F401  (registers all timm models)
from growing_nn import utils
from growing_nn.cli.args import get_args_parser
from growing_nn.data import RASampler, build_dataset, new_data_aug_generator
from growing_nn.growth import growth_wrapper, remove_garbage, split_nodewise
from growing_nn.training import DistillationLoss, evaluate, train_one_epoch


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(args):
    warnings.filterwarnings("ignore")
    utils.init_distributed_mode(args)

    print(args)

    config = {
        "architecture": "deit-tiny-growth-step-lr",
        "baseline": "False",
        "dataset": "ImageNet-100",
        "epochs": 300,
    }
    if args.wandb != "":
        utils.init_wandb("growing-nn-new", args.wandb, config)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    seed_everything(42)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    x, y = next(iter(data_loader_val))
    print("X shape", x.shape)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
    )

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    if args.attn_only:
        for name_p, p in model.named_parameters():
            if ".attn." in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except AttributeError:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except AttributeError:
            print("no position encoding")
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except AttributeError:
            print("no patch embed")

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else "", resume=""
        )

    print(model)
    model = growth_wrapper(model)
    if args.resume_growth != "":
        model = torch.load(args.resume_growth)
        print(model)
        model.train()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    base_lr = args.lr
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        base_lr = linear_scaled_lr
        args.lr = linear_scaled_lr
    print("Learning Rate", args.lr)
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    teacher_model = None
    if args.distillation_type != "none":
        assert args.teacher_path, "need to specify teacher-path when using distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool="avg",
        )
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.teacher_path, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location="cpu")
        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        del model, model_without_ddp, optimizer, lr_scheduler, loss_scaler
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint = torch.load(args.resume, map_location=torch.device(device))
        model_without_ddp = checkpoint["model"]
        model_without_ddp.train()
        model_without_ddp.to(device)
        optimizer = checkpoint["optimizer"]
        lr_scheduler = checkpoint["lr_scheduler"]
        args.start_epoch = checkpoint["epoch"]
        loss_scaler = checkpoint["scaler"]
        del checkpoint
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module

    print(sum([p.numel() for p in model.parameters()]))
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    split_count = 0
    cn = 0
    print("Utilization after Initialization")
    GPUtil.showUtilization()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(data_loader_val, model, device)
        print(train_stats.keys(), test_stats.keys())
        model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if args.wandb != "":
            utils.log_wandb(
                {
                    "train_lr": train_stats["lr"],
                    "train_loss": train_stats["loss"],
                    "test_loss": test_stats["loss"],
                    "test_acc1": test_stats["acc1"],
                    "test_acc5": test_stats["acc5"],
                    "model_params": n_parameters,
                }
            )

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if utils.is_main_process():
                os.makedirs(args.cp, exist_ok=True)
                utils.save_on_master(
                    {
                        "model": model_without_ddp,
                        "optimizer": optimizer,
                        "lr_scheduler": lr_scheduler,
                        "epoch": epoch,
                        "scaler": loss_scaler,
                        "args": args,
                    },
                    f"{args.cp}/{cn}_{int(max_accuracy)}.pt",
                )
                cn += 1

        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        print("Utilization after Epoch")
        GPUtil.showUtilization()

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if args.split_epochs != -1 and epoch >= args.initwarm:
            if epoch != 0 and (epoch - args.initwarm) % args.split_epochs == 0:
                print("Splitting")
                print(args.top_percent, args.layer_percent)
                p_bud = args.param_budget * (train_stats["lr"] / base_lr)
                if utils.is_main_process():
                    print("Utilization Before Splitting")
                    GPUtil.showUtilization()

                    split_nodewise(model_without_ddp, optimizer, p_bud, epoch, args.top_percent, args.split_warmup)
                    model_without_ddp.to(device)

                    print("Utilization after Splitting")
                    GPUtil.showUtilization()

                    split_count += 1
                    print(model_without_ddp)
                    os.makedirs(args.folder, exist_ok=True)
                    utils.save_on_master(
                        {
                            "model": model_without_ddp,
                            "optimizer": optimizer,
                            "lr_scheduler": lr_scheduler,
                            "epoch": epoch,
                            "scaler": loss_scaler,
                            "args": args,
                        },
                        f"{args.folder}/{split_count}.pt",
                    )
                    print(f"World Size = {utils.get_world_size()}")
                    for r in range(utils.get_world_size()):
                        utils.save_on_master(
                            {
                                "model": model_without_ddp,
                                "optimizer": optimizer,
                                "lr_scheduler": lr_scheduler,
                                "epoch": epoch,
                                "scaler": loss_scaler,
                                "args": args,
                            },
                            f"{args.folder}/chk_{r}.pt",
                        )

                torch.distributed.barrier()
                if not utils.is_main_process():
                    remove_garbage(model_without_ddp)
                torch.distributed.barrier()

                model_without_ddp.to("cpu")
                del model, model_without_ddp, optimizer, lr_scheduler, loss_scaler
                checkpoint = torch.load(f"{args.folder}/chk_{utils.get_rank()}.pt", map_location=torch.device(device))
                model_without_ddp = checkpoint["model"]
                model_without_ddp.train()
                model_without_ddp.to(device)
                optimizer = checkpoint["optimizer"]
                lr_scheduler = checkpoint["lr_scheduler"]
                loss_scaler = checkpoint["scaler"]
                print(model_without_ddp)
                print("here")
                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(
                        model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True
                    )
                    model_without_ddp = model.module

                print("reached_here")
                print(f"Rank {utils.get_rank()} Successful")
                torch.distributed.barrier()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Split Count = {split_count}")
                print("CPU usage", psutil.cpu_percent(5))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def main_cli():
    """Console-script entry point (``growing-nn-train-growth``)."""
    parser = argparse.ArgumentParser("DeiT training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


if __name__ == "__main__":
    main_cli()
