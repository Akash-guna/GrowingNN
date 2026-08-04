"""
growing_nn.growth
==================

The "node growth" mechanism: dynamically splitting saturated linear
layers of a transformer during training. Split, by concern, out of the
original monolithic top-level ``GrowthNew.py`` / ``growth_utils_node_new.py``:

- :mod:`growing_nn.growth.block`     - :class:`GrowthModel` /
  :func:`GrowthBlock`, the module that recombines an "old" and "new"
  branch of a split layer.
- :mod:`growing_nn.growth.layers`    - locating Linear layers & renumbering
  architecture trees.
- :mod:`growing_nn.growth.eigen`     - per-neuron eigenvalue/saturation
  analysis.
- :mod:`growing_nn.growth.selection` - deciding which layers/neurons to
  split under a parameter budget.
- :mod:`growing_nn.growth.mutate`    - the depth/width growth operators
  that actually create new layers.
- :mod:`growing_nn.growth.schedule`  - top-level orchestration
  (``growth_wrapper``, ``split_nodewise``, ``remove_garbage``) called
  from the training loop.

Everything is re-exported here so ``from growing_nn import growth;
growth.split_nodewise(...)`` (or ``from growing_nn.growth import
split_nodewise``) works the same way ``from growth_utils_node_new import
*`` used to.
"""

from growing_nn.growth.block import GrowthBlock, GrowthModel
from growing_nn.growth.eigen import (
    calc_all_eigs,
    calculate_eig,
    calculate_min_eig,
    layer_negative,
    plot_eig,
    ret_flattened,
    split_matrix,
)
from growing_nn.growth.layers import (
    assign_model,
    get_all_linear_layers,
    get_all_linear_layers_transformer,
    ret_growth_model,
    return_arc_array,
)
from growing_nn.growth.mutate import (
    create_layer_width,
    create_new_layer_new,
    create_width_growth,
    update_next_layer_weights,
    update_qk_model_width,
)
from growing_nn.growth.schedule import convert, growth_wrapper, remove_garbage, split_nodewise
from growing_nn.growth.selection import find_split_layers_param_quota, get_num_layers_below

__all__ = [
    "GrowthBlock",
    "GrowthModel",
    "calc_all_eigs",
    "calculate_eig",
    "calculate_min_eig",
    "layer_negative",
    "plot_eig",
    "ret_flattened",
    "split_matrix",
    "assign_model",
    "get_all_linear_layers",
    "get_all_linear_layers_transformer",
    "ret_growth_model",
    "return_arc_array",
    "create_layer_width",
    "create_new_layer_new",
    "create_width_growth",
    "update_next_layer_weights",
    "update_qk_model_width",
    "convert",
    "growth_wrapper",
    "remove_garbage",
    "split_nodewise",
    "find_split_layers_param_quota",
    "get_num_layers_below",
]
