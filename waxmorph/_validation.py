from numbers import Integral


def _validate_same_shape(name, pred_shape, target_shape) -> None:
    pred_shape, target_shape = tuple(pred_shape), tuple(target_shape)
    if pred_shape != target_shape:
        raise ValueError(f"{name} requires the same shape, got {pred_shape} and {target_shape}")


def _validate_chamfer_shapes(pred_shape, target_shape) -> None:
    pred_shape, target_shape = tuple(pred_shape), tuple(target_shape)
    if len(pred_shape) != 2 or len(target_shape) != 2:
        raise ValueError(
            f"chamfer_distance requires rank-2 inputs, got {pred_shape} and {target_shape}"
        )
    if 0 in pred_shape or 0 in target_shape:
        raise ValueError(
            f"chamfer_distance requires nonempty inputs, got {pred_shape} and {target_shape}"
        )
    if pred_shape[1] != target_shape[1]:
        raise ValueError(
            f"chamfer_distance requires equal feature width, got {pred_shape} and {target_shape}"
        )


def _validate_mlp_config(activation, choices, num_layers) -> int:
    if activation not in choices:
        raise ValueError(f"Unknown activation '{activation}'. Choose from {list(choices)}.")
    if isinstance(num_layers, bool) or not isinstance(num_layers, Integral):
        raise TypeError("num_layers must be a non-boolean integer")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    return int(num_layers)
