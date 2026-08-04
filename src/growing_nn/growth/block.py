"""
The core growth block: :class:`GrowthModel` recursively wires together an
"old" branch and a "new" (split-off) branch of linear layers, recombining
them through a GELU-gated sum, plus a small custom autograd
``saved_tensors_hooks`` pair (:class:`SharedSaveTensor` /
:func:`pack_hook` / :func:`unpack_hook`) that de-duplicates identical
tensors saved for backward across the (potentially many) growth blocks in
a model, to reduce activation-memory overhead.

Split out of the original top-level ``GrowthNew.py`` with no functional
changes.
"""
import torch
import torch.nn as nn


class SharedSaveTensor:
    """De-duplicating store for tensors saved by autograd for backward.

    Used as the "packed" representation in a
    ``torch.autograd.graph.saved_tensors_hooks`` context (see
    :func:`pack_hook` / :func:`unpack_hook`): rather than saving the same
    activation tensor once per growth block that references it, identical
    tensors are stored once and looked up by content.
    """

    # tensor_dic with base64 string representation of tensor as key and tensor as value
    created_objects = 0
    base64_tensor = {}

    def __init__(self, sum=None, index=None):
        self.sum = sum
        self.index = index
        SharedSaveTensor.created_objects += 1

    def search_for_tensor(self, tensor):
        if self.sum in SharedSaveTensor.base64_tensor.keys():
            arr = SharedSaveTensor.base64_tensor[self.sum]
            for i in range(len(arr)):
                if torch.equal(arr[i], tensor):
                    self.index = i
                    return i
            return "NO_TENSOR"
        else:
            return "NO_KEY"

    def store_tensor(self, tensor):
        self.sum = str(torch.mean(torch.tensor([1.0, 1.0, 1.0])).item())
        if self.search_for_tensor(tensor) == "NO_KEY":
            SharedSaveTensor.base64_tensor[self.sum] = [tensor]
            self.index = 0
        elif self.search_for_tensor(tensor) == "NO_TENSOR":
            SharedSaveTensor.base64_tensor[self.sum].append(tensor)
            self.index = len(SharedSaveTensor.base64_tensor[self.sum]) - 1
        else:
            pass
        return [self.sum, self.index]

    def load_tensor(self):
        return SharedSaveTensor.base64_tensor[self.sum][self.index]

    def __del__(self):
        SharedSaveTensor.created_objects -= 1
        if SharedSaveTensor.created_objects == 0:
            SharedSaveTensor.base64_tensor = {}


def pack_hook(tensor):
    save_tensor_obj = SharedSaveTensor()
    save_tensor_obj.store_tensor(tensor)
    return save_tensor_obj


def unpack_hook(save_tensor_obj):
    tensor = save_tensor_obj.load_tensor()
    return tensor


class GrowthModel(nn.Module):
    """The Class to Create The Growing NN Block.

    Input:
        layer_array: A List containing all dense layers ordered from top
            to bottom

        act_on: Specifies whether to have an activation (used to turn off
            activation if the block is followed by a GELU in DeiT)

        architecture_array: A Tree with 5 children per parent (each child
            denotes one layer [old_layer,new_layer,feature_bottleneck,
            old_split,skip]) with value of each child could be a tree or
            0. 0 denotes a leaf node which helps to add the layer to the
            model.
                Eg [0,0,0] -> a FeedForward Network with 3 Linear Layers.
                   [0,[[0,0],perm],0] -> Linear Layer -> Growth Block ->
                   Linear Layer

                perm -> is the order to shuffle the concatenation of
                old_split and split neurons to get the pre-split order.
                [1,3,4] -> not split, [0,2] -> split
                [1,3,4,0,2] -> after concat, perm=[3,0,4,1,2],
                after shuffle = [0,1,2,3,4].
                Perm is added to each level of the tree (the level of the
                tree is a growth block).

    Working: Layer array contains the layer, architecture array maps the
        layers to its corresponding position in a Neural Network.

    Output: A GrowthModel object (a Model).
    """

    def __init__(self, layer_array, architecture_array, act_on=True):
        super(GrowthModel, self).__init__()
        self.layer_dict = nn.ModuleDict(layer_array)
        self.architecture_array = architecture_array
        self.layer_count = 0
        self.act_on = act_on
        self.gelu = nn.GELU()

    def module(self, arc_array, x):
        """Growth Block Definition. Recursively calls itself for
        branching.

        Input:
            arc_array : architecture array -> Main Call. A child
                architecture array of parent if recursively called
            x : input (from forward())
        """
        # The architecture_array here has a fixed size of 5 [a,a] a = another arcitecture_array | 0
        architecture_array = arc_array[0]
        perm = arc_array[1]

        if architecture_array[0] == 0:
            # old_layer
            x1 = self.layer_dict[str(self.layer_count)](x)
            self.layer_count += 1
        else:
            # old_layer
            x1 = self.module(architecture_array[0], x)
        if architecture_array[1] == 0:
            # new_layer
            x2 = self.layer_dict[str(self.layer_count)](x)
            self.layer_count += 1
        else:
            # new_layer
            x2 = self.module(architecture_array[1], x)

        mid = int(x2.shape[-1] / 2)
        if len(x2.shape) == 2:
            # Handling Normal Linear Layers
            gelu_in = x2[:, :mid] + x2[:, mid:]
            x1[:, perm] += self.gelu(gelu_in) + x2[:, :mid] + x2[:, mid:]
        else:
            # Handling QKV Layers, Stacked Linear Layers in Transformers
            gelu_in = x2[:, :, :mid] + x2[:, :, mid:]
            x1[:, :, perm] += self.gelu(gelu_in) + x2[:, :, :mid] + x2[:, :, mid:]
        return x1

    def forward(self, x):
        self.layer_count = 0
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            for i in range(len(self.architecture_array) - 1):
                # If a linear layer, add to model graph
                if self.architecture_array[i] == 0:
                    x = self.gelu(self.layer_dict[str(self.layer_count)](x))
                    self.layer_count += 1
                else:
                    # Else call module() to handle growth
                    x = self.gelu(self.module(self.architecture_array[i], x))

            # For final layer
            if self.architecture_array[len(self.architecture_array) - 1] == 0:
                if self.act_on:
                    x = self.gelu(self.layer_dict[str(self.layer_count)](x))
                else:
                    x = self.layer_dict[str(self.layer_count)](x)
            else:
                x = self.module(self.architecture_array[len(self.architecture_array) - 1], x)
        self.layer_count = 0
        return x


def GrowthBlock(linear, act_on=False):
    """Wraps a linear layer in a linear model with a GrowthModel class."""
    layer_array = {"0": linear}
    # Since single linear layer arc arr = [0]
    arc_array = [0]
    gb = GrowthModel(layer_array, arc_array, act_on)
    return gb
