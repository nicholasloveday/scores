"""
Implementation of the probabilistic Fractions Skill Score (pFSS) for ensemble forecasts.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import xarray as xr

from scores.spatial.fss_impl import (
    _make_numpy_threshold_operator,
    fss_2d_binary,
)
from scores.typing import FlexibleDimensionTypes
from scores.utils import gather_dimensions


def pfss_for_ensemble(  # pylint: disable=too-many-locals,too-many-arguments
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    event_threshold: float,
    window_size: Tuple[int, int],
    spatial_dims: Tuple[str, str],
    ensemble_member_dim: str,
    zero_padding: bool = False,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    threshold_operator: Optional[Callable] = np.greater,
) -> xr.DataArray:
    """
    Calculates the probabilistic Fractions Skill Score (pFSS) for an ensemble forecast.

    The pFSS is the Fractions Skill Score (FSS) calculated from the ensemble probability of an event,
    rather than from a single deterministic forecast. At each grid point, the ensemble probability is the
    proportion of ensemble members that forecast the event, as defined by ``event_threshold`` and
    ``threshold_operator``. Within each window, the mean of the ensemble probability is the neighbourhood
    ensemble probability (NEP), which is compared with the fraction of observed events in the
    same window:

    .. math::

        \\text{pFSS} = 1 - \\frac{\\sum_w (P_w - O_w)^2}{\\sum_w P_w^2 + \\sum_w O_w^2}

    where :math:`P_w` is the NEP and :math:`O_w` is the observed event fraction in window :math:`w`, and the
    sums are over all window positions. If other dimensions are being reduced, the terms are accumulated over
    those dimensions before the final calculation, in the same way as :py:func:`scores.spatial.fss_2d`.

    .. note::
        **NaN handling:** NaNs are never passed to the FSS calculation, because the summed-area tables used
        to calculate the window totals would spread a single NaN to every window below and to the right of it.
        Instead:

        - NaNs in ``fcst`` are ignored when calculating the ensemble probability, i.e. the probability is the
          proportion of the *non-missing* ensemble members that forecast the event. Ensembles with a varying
          number of valid members are therefore supported.
        - A grid point is treated as missing if all ensemble members are NaN in ``fcst``, or if ``obs`` is NaN.
          Missing grid points are set to be non-events (a probability of 0 in ``fcst`` and an event value of 0
          in ``obs``), so they add nothing to the window fractions of either field. Windows that overlap
          missing grid points are still included in the score.
        - If there is no valid data at all over the dimensions being reduced, e.g. a time step where ``fcst`` is
          entirely NaN and ``preserve_dims="time"``, the score at that point is NaN.

    Args:
        fcst: An array of forecasts. Must contain ``spatial_dims`` and ``ensemble_member_dim``.
        obs: An array of observations (same spatial shape as ``fcst``)
        event_threshold: A scalar to compare ``fcst`` and ``obs`` fields to generate a
            binary "event" field. (defaults to ``np.greater``).
        window_size: A pair of positive integers ``(height, width)`` of the sliding
            window_size; the window size must be greater than 0 and fit within
            the shape of ``obs`` and ``fcst``.
        spatial_dims: A pair of dimension names ``(x, y)`` where ``x`` and ``y`` are
            the spatial dimensions to slide the window along to compute the FSS.
            e.g. ``("lat", "lon")``.
        ensemble_member_dim: The dimension name of the ensemble member. This dimension must be in
            ``fcst`` and must not be in ``obs``.
        zero_padding: If set to true, applies a 0-valued window border around the
            data field before computing the FSS. If set to false (default), it
            uses the edges (thickness = window size) of the input fields as the border.
            One can think of it as:

            - zero_padding = False => inner border
            - zero_padding = True => outer border with 0 values
        reduce_dims: Optional set of additional dimensions to aggregate over
            (mutually exclusive to ``preserve_dims``).
        preserve_dims: Optional set of dimensions to keep (all other dimensions
            are reduced). Cannot preserve spatial dimensions.
        threshold_operator: The threshold operator used to generate the binary
            event field. E.g. ``np.greater``. Note: this may depend on the backend
            ``compute method`` and not all operators may be supported. Generally,
            ``np.greater``, ``np.less`` and their counterparts. Defaults to "np.greater".

    Returns:
        An ``xarray.DataArray`` containing the pFSS computed over the ``spatial_dims``.

        The resultant array will have the score aggregated over the dimensions specified by
        ``reduce_dims`` / ``preserve_dims``. ``ensemble_member_dim`` is never present in the output.
        If neither is specified, all dimensions are reduced and a single value is returned.
        The score takes values between 0 and 1, where 1 is a perfect score. If no events are forecast
        or observed in the data being aggregated, the score is NaN.

    Raises:
        DimensionError: If the window size is larger than the input arrays or is not a pair of
            positive integers, or if ``spatial_dims`` are not a pair of dimensions present in both
            ``fcst`` and ``obs``.

        ValueError: If ``ensemble_member_dim`` is not in ``fcst.dims``, if it is in ``obs.dims``, ``reduce_dims``
            or ``preserve_dims``, or if ``threshold_operator`` is not a supported operator.

        DimensionWarning: If ``spatial_dims`` are attempting to be preserved e.g. in ``preserve_dims``

    See Also:
        - :py:func:`scores.spatial.fss_2d`
        - :py:func:`scores.spatial.fss_2d_binary`

    References:
        1. Necker, T., Wolfgruber, L., Kugler, L., Weissmann, M., Dorninger, M., & Serafin, S. (2024).
           The fractions skill score for ensemble forecast verification. Quarterly Journal of the
           Royal Meteorological Society, 150(764), 4457–4477. https://doi.org/10.1002/qj.4824.

    Examples:
        >>> import xarray as xr
        >>> from scores.probability import pfss_for_ensemble

        >>> # Create a simple 2-member ensemble forecast and observations on a 3x3 grid
        >>> fcst = xr.DataArray(
        ...     [
        ...         [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        ...         [[0, 0, 2], [1, 3, 3], [1, 3, 4]],
        ...     ],
        ...     dims=["member", "x", "y"],
        ... )

        >>> obs = xr.DataArray([[0, 0, 1], [1, 2, 2], [2, 2, 3]], dims=["x", "y"])

        >>> # Compute pFSS with threshold=1.5 and 2x2 window
        >>> pfss_for_ensemble(
        ...     fcst,
        ...     obs,
        ...     event_threshold=1.5,
        ...     window_size=(2, 2),
        ...     spatial_dims=("x", "y"),
        ...     ensemble_member_dim="member",
        ... )
        <xarray.DataArray ()> Size: 8B
        array(0.97991968)

    """
    dims_to_reduce = gather_dimensions(
        fcst.dims,
        obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=ensemble_member_dim,
    )

    np_thrsh_op = _make_numpy_threshold_operator(threshold_operator).get()

    fcst_prob = np_thrsh_op(fcst, event_threshold).where(fcst.notnull()).mean(ensemble_member_dim)
    obs_events = np_thrsh_op(obs, event_threshold).where(obs.notnull())
    valid = fcst_prob.notnull() & obs_events.notnull()
    fcst_prob = fcst_prob.where(valid, 0.0)
    obs_events = obs_events.where(valid, 0.0)

    pfss = fss_2d_binary(
        fcst_prob,
        obs_events,
        window_size=window_size,
        spatial_dims=spatial_dims,
        check_boolean=False,
        zero_padding=zero_padding,
        reduce_dims=dims_to_reduce,
    )
    # Mask results where data was missing.
    # Also, where there are no events in either forecast or observation, mask the result
    # since the fraction in pFSS would be undefined.
    reduced = [d for d in valid.dims if d not in pfss.dims]
    any_event = ((fcst_prob > 0) | (obs_events > 0)).any(dim=reduced)
    pfss = pfss.where(any_event)
    return pfss
