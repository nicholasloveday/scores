"""
Functions related to aggregating data
"""

import warnings
from typing import Optional

import numpy as np
import xarray as xr

from scores.processing.matching import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_weights


SUPPORTED_METHODS_STR = ["mean", "sum"]


def _aggregate_error_builder(name, error_type=Exception):
    """
    Generic builder for various input error types
    """
    return type(name, (error_type,), {})


AggregateError_InputKey = _aggregate_error_builder("InputKeyError", KeyError)

AggregateError_InputValue = _aggregate_error_builder("InputValueError", ValueError)

AggregateError_InputType = _aggregate_error_builder("InputTypeError", TypeError)

AggregateError_Compute = _aggregate_error_builder("ComputeError", ValueError)

AggregateError_Critical = _aggregate_error_builder("CriticalError", RuntimeError)


# black is not setup to format long strings properly
# fmt: off

# usage: raise ERROR_UNREACHABLE
ERROR_UNREACHABLE = AggregateError_Compute(
    "CRITICAL FAILURE! Unreachable code, please raise a github ticket quoting "
    "any traceback logs."
)

# usage: raise ERROR_INVALID_METHOD("agg")
ERROR_INVALID_METHOD = lambda method: AggregateError_InputValue(
    "Method must be one of {}, got '{}'".format(SUPPORTED_METHODS_STR, method)
)

# usage: raise ERROR_WEIGHT_TYPE_MISMATCH
ERROR_WEIGHT_TYPE_MISMATCH = AggregateError_InputType(
    "`weights` cannot be an xr.Dataset when `values` is an xr.DataArray"
)

# usage: raise ERROR_UNSPECIFIED_WEIGHTS_FOR_VARIABLE("var")
ERROR_UNSPECIFIED_WEIGHTS_FOR_VARIABLE = lambda var_name: AggregateError_InputKey(
    "No weights provided for variable '{}'".format(var_name)
)

# usage: warnings.warn(WARN_WEIGHTS_IGNORED_STR)
WARN_WEIGHTS_IGNORED_STR = (
    "Weights were provided but the point-wise score across all dimensions is "
    "being preserved.\nWeights will be ignored."
)

# fmt: on


def aggregate(
    values: XarrayLike,
    *,
    reduce_dims: FlexibleDimensionTypes | None,
    weights: Optional[XarrayLike] = None,
    method: str = "mean",
) -> XarrayLike:
    """
    Computes a weighted or unweighted aggregation of the input data across specified dimensions.
    The input data is typically the "score" at each point.

    This function applies a mean reduction or a sum over the dimensions given by ``reduce_dims`` on
    the input ``values``, optionally using weights to compute a weighted mean. Weighting
    is performed using xarray's `.weighted()` method. The `method` arg specifies if you
    want to produce a weighted mean or weighted sum.

    If `reduce_dims` is None, no aggregation is performed and the original `values` are
    returned unchanged.

    If `weights` is None, an unweighted mean is computed. If weights are provided, negative
    weights are not allowed and will raise a `ValueError`.

    If weights are provided but `reduce_dims` is None (i.e., no reduction), a `UserWarning`
    is emitted since the weights will be ignored.

    Weights must not contain NaN values. Missing values can be filled by ``weights.fillna(0)``
    if you would like to assign a weight of zero to those points (e.g., masking).

    Args:
        values: Input data to be reduced. Typically an `xr.DataArray` or `xr.Dataset`.
        reduce_dims: Dimensions over which to apply the mean. Can be a string, list of
            strings, or None. If None, no reduction is performed.
        weights: Weights to apply for weighted averaging.
            Must be broadcastable to `values` and contain no negative values. If None,
            an unweighted mean is calculated. Defaults to None.
        method: Aggregation method to use. Either "mean" or "sum". Defaults to "mean".

    Returns:
        An xarray object (same type as the input) with (un)weighted mean or sum of ``values``

    Raises:
        ValueError: If `weights` contains any negative values.
        ValueError: if `weights` contains any NaN values
        ValueError: if `method` is not 'mean' or 'sum'
        ValueError: if `weights` is an xr.Dataset when `values` is an xr.DataArray
        NotImplementedError: if `method` is sum and weights is an xr.Dataset

    Warnings:
        UserWarning: If weights are provided but no reduction is performed (`reduce_dims` is None),
        a warning is issued since weights are ignored.

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> da = xr.DataArray(np.arange(6).reshape(2, 3), dims=['x', 'y'])
        >>> weights = xr.DataArray([1, 2], dims=['x'])
        >>> apply_weighted_mean(da, reduce_dims=['x'], weights=weights)
        <xarray.DataArray (y: 3)>
        array([2., 3., 4.])
        Dimensions without coordinates: y

    """
    # bypass any computation if no dimensions are being reduced. (identity function)
    if reduce_dims is None:
        # warn user if weights are provided without any reduction specified.
        if weights is not None:
            warnings.warn(WARN_WEIGHTS_IGNORED_STR)
        return values

    _check_aggregate_inputs(values, reduce_dims, weights, method)

    match method:
        case "mean":
            if weights is not None:
                return _weighted_mean(values, weights, reduce_dims)
            return values.mean(reduce_dims)
        case "sum":
            if weights is not None:
                return _weighted_sum(values, weights, reduce_dims)
            return values.sum(reduce_dims)
        case _:
            # already checked in _check_aggregate_inputs
            raise ERROR_UNREACHABLE


def raise_if_invalid_aggregation_method(method: str):
    if (not isinstance(method, str)) or (method not in SUPPORTED_METHODS_STR):
        raise ERROR_INVALID_METHOD(method)


def _weighted_mean(
    values: XarrayLike,
    weights: XarrayLike,
    reduce_dims: FlexibleDimensionTypes,
) -> XarrayLike:
    """
    Calculates the weighted mean of `values` using `weights` over specified dimensions.

    xarray doesn't allow ``.weighted`` to take ``xr.Dataset`` as weights, so we need to do it ourselves
    """
    # safety
    assert reduce_dims is not None
    assert weights is not None

    if isinstance(weights, xr.Dataset):
        w_results = {}
        for name, da in values.data_vars.items():
            w = weights[name]
            da_aligned, w_aligned = broadcast_and_match_nan(da, w)

            # `check_weights` in `_check_aggregate_inputs` ensures that `weights`
            # has at least one positive value and will raise an error.
            # However, if a value in w_aligned.sum(dim=reduce_dims) is zero,
            # a NaN will be produced for that point.
            w_numerator = (da_aligned * w_aligned).sum(dim=reduce_dims)
            w_denominator = w_aligned.sum(dim=reduce_dims)
            w_results[name] = w_numerator / w_denominator

        return xr.Dataset(w_results)

    values = values.weighted(weights)

    return values.mean(reduce_dims)


def _weighted_sum(
    values: XarrayLike,
    weights: XarrayLike,
    reduce_dims: FlexibleDimensionTypes,
) -> XarrayLike:
    """
    Calculated the weighted sum of `values` using `weights` over specified dimensions.

    FUTUREWORK:
        - xr.dot uses einsum, which has an "optimized" mode, which apparently
          speeds things up a lot - research this.
        - add opt_einsum as a dependency for optimized summations.
    """
    # safety: checked in aggregate()
    assert reduce_dims is not None

    def _align_and_fillzero(_v, _w, _dims):
        fn_nan = xr.ufuncs.isnan
        v_aligned, w_aligned = broadcast_and_match_nan(_v, _w)

        # zero out masked values since they don't need to be summed
        result = (v_aligned.fillna(0), w_aligned.fillna(0))

        # get reduced mask to preserve nans, post computation
        # NOTE: only need to do this for v_aligned, since w_aligned is matched.
        nan_mask = xr.ufuncs.isnan(v_aligned).all(dim=_dims)

        return result, nan_mask

    def _reduce_sum(_v, _w, _dims):
        (_v, _w), nan_mask = _align_and_fillzero(_v, _w, _dims)
        result = xr.dot(_v, _w, dim=_dims)
        return result.where(~nan_mask, np.nan)

    if isinstance(values, xr.Dataset):
        w_results = {}
        w = weights  # assume weights are arrays/dataarrays

        for name, da in values.data_vars.items():
            # if weights are actually datasets, attempt to extract the
            # appropriate variable
            if isinstance(weights, xr.Dataset):
                # ---
                # safety: this is already checked in _check_aggregate_inputs;
                # if 'weights' is a dataset, it cannot be broadcast to an
                # unspecified variable in 'values' and vice-versa, because a
                # concept of a "default" does not exist.
                if name not in weights:
                    raise ERROR_UNREACHABLE
                # ---
                w = weights[name]

            w_results[name] = _reduce_sum(da, w, reduce_dims)

        return xr.Dataset(w_results)

    # safety: these are the only viable options
    assert isinstance(values, xr.DataArray)
    assert not isinstance(weights, xr.Dataset)

    return _reduce_sum(values, weights, reduce_dims)


def _check_aggregate_inputs(
    values: XarrayLike, reduce_dims: FlexibleDimensionTypes | None, weights: XarrayLike | None, method: str,
):
    """
    This function checks the inputs to the aggregate function.

    It checks that:
    - `method` is either 'mean' or 'sum'
    - `weights` does not contain negative values
    - `weights` does not contain NaN values
    - `weights` were provided, `reduce_dims` is not None
    - `weights` is not an xr.Dataset when `values` is an xr.DataArray
    - if `values is an xr.Dataset`, and `weights` is an xr.Dataset, it must have the same variables

    Args:
        values: The input data to be reduced in :py:func:`aggregate`.
        reduce_dims: The dimensions over which to apply the mean in :py:func:`aggregate`.
        weights: The weights to apply for weighted averaging in :py:func:`aggregate`.
        method: The aggregation method to use, either "mean" or "sum" in :py:func:`aggregate`.
    """
    is_dataset = lambda maybe_ds: isinstance(maybe_ds, xr.Dataset)

    raise_if_invalid_aggregation_method(method)

    if weights is not None:
        check_weights(weights)

        if reduce_dims is not None:
            # ---
            # weights cannot have more structural information than values
            # i.e.
            # weights = dataarray, values = dataset - OK
            # weights = dataset, values = dataarray or numpy array - NOT OK
            # if they are the same type - OK
            if is_dataset(weights) and not is_dataset(values):
                raise ERROR_WEIGHT_TYPE_MISMATCH
            # ---

            # ---
            # if both `values` and `weights` are datasets,
            # check that all variables in `values` are present in `weights`,
            # otherwise `weights` is underspecified, and therefore the weighted
            # aggregation is ambiguous.
            if is_dataset(weights) and is_dataset(values):
                for name in values.data_vars:
                    if name not in weights:
                        raise ERROR_UNSPECIFIED_WEIGHTS_FOR_VARIABLE(name)
            # ---
