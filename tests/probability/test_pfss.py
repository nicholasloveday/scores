"""
Tests for the probabilistic Fractions Skill Score (pFSS) implementation.
"""

import pytest
import xarray as xr

from scores import sample_data as sd
from scores.probability import pfss_for_ensemble
from scores.spatial.fss_impl import fss_2d


# TODO add test for zero_padding and threshold_operator
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


# 2 members, 1x1 window
# 2 members, 2x2 window
def test_multiple_ens_members():
    """
    Test that the pFSS for multiple ensemble members returns a valid score.
    """
    pass


def test_identical_mems_equal_one_mem():
    """
    Test that identical ensemble members produce the same pFSS as a single-member ensemble.
    """
    pass


# Test all with something like `preserve_dims="time"` and `reduce_dims="time"`
# Test all NaNs in one ensemble member at one point
# Test one member NaN at one point
# Test obs NaN at one point
# Everything is NaN at all points
def test_pffs_nan_handling():
    """
    Test that the pFSS correctly handles NaN values in the forecast and observation data.
    """
    pass


# Test ens mem dim not in fcst
# Test ens mem dim in obs
# Test ens mem dim in reduce dims
# Test ens mem dim in preserve dims
# Test threshold_operator not supported
def test_pffs_raises():
    """
    Test that the pFSS raises appropriate errors for invalid inputs.
    """


# Test Dimension warning
def test_pffs_warns():
    """
    Test that the pFSS raises appropriate warnings.
    """
    pass
