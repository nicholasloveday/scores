"""
Tests for the probabilistic Fractions Skill Score (pFSS) implementation.
"""

import pytest
import xarray as xr

from scores import sample_data as sd
from scores.probability import pfss_for_ensemble
from scores.spatial.fss_impl import fss_2d


@pytest.mark.parametrize(
    (
        "window_size",
        "event_threshold",
        "reduce_dims",
        "preserve_dims",
    ),
    [
        ((2, 2), 2.0, None, None),
        ((2, 2), 8.0, None, None),
        ((5, 5), 2.0, None, None),
        ((5, 5), 8.0, None, None),  # output_dims: scalar
        ((2, 2), 5.0, ["time"], None),  # output_dims: 1 x lead_time
        ((2, 2), 5.0, None, ["time"]),  # output_dims: 1 x time
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
        ),  # output_dims: time x lead_time
    ],
)
def test_pfss_equals_fss_for_single_member(window_size, event_threshold, reduce_dims, preserve_dims):
    """
    Test that the pFSS for a single-member ensemble equals the standard FSS.

    These tests inputs were taken from scores.testing.spatial.test_fss_2d_binary
    """
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    fcst_ens = da_fcst.expand_dims(ens_mem=[1])

    pfss_score = pfss_for_ensemble(
        fcst_ens,
        da_obs,
        event_threshold=event_threshold,
        window_size=window_size,
        ensemble_member_dim="ens_mem",
        spatial_dims=("lat", "lon"),
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    fss_score = fss_2d(
        da_fcst,
        da_obs,
        event_threshold=event_threshold,
        window_size=window_size,
        spatial_dims=("lat", "lon"),
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    xr.testing.assert_equal(pfss_score, fss_score)
