"""
Tests for the probabilistic Fractions Skill Score (pFSS) implementation.

Note that the underlying FSS code that pFSS calls is thoroughly tested in
tests/spatial/test_fss.py.
"""

import numpy as np
import pytest
import xarray as xr

from scores import sample_data as sd
from scores.probability import pfss_for_ensemble
from scores.spatial.fss_impl import fss_2d
from scores.utils import DimensionError
import scores.spatial

FCST = xr.DataArray(
    [
        # member 1
        [[0, 3, 3, 0], [0, 3, 2, 0], [4, 0, 0, 0], [4, 4, 0, 0]],
        # member 2
        [[0, 3, 0, 0], [0, 2, 2, 0], [4, 0, 0, 3], [0, 4, 0, 0]],
        # member 3
        [[0, 0, 0, 0], [3, 0, 2, 0], [0, 0, 50, 3], [0, 3, 0, -9]],
        # member 4
        [[0, 0, 0, 100], [0, 0, 0, 0], [0, 0, 3, 3], [0, 0, 0, 3]],
    ],
    dims=["ens_mem", "x", "y"],
    coords={"ens_mem": [1, 2, 3, 4], "x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
)
OBS = xr.DataArray(
    [[0, 3, 3, 0], [0, 3, 2, 0], [4, 0, 0, 0], [4, 4, 0, 0]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
)
FCST_SINGLE_NAN = FCST.astype(float).copy()
FCST_SINGLE_NAN.loc[{"ens_mem": 2, "x": 1, "y": 1}] = np.nan

FCST_MEM_1_NAN = FCST.astype(float).copy()
FCST_MEM_1_NAN.loc[{"ens_mem": 1}] = np.nan

FCST_POINT_ALL_NAN = FCST.astype(float).copy()
FCST_POINT_ALL_NAN.loc[{"x": 1, "y": 1}] = np.nan

FCST_PERFECT = OBS.expand_dims(ens_mem=[1, 2, 3, 4])

FCST_EXTRA_DIM = FCST.expand_dims(time=[1, 2])

FCST_EXTRA_DIM_NAN = FCST_EXTRA_DIM.astype(float).copy()
FCST_EXTRA_DIM_NAN.loc[{"time": 1}] = np.nan

FCST_RENAMED_DIM = FCST.rename({"x": "lat", "y": "lon"})
OBS_RENAMED_DIM = OBS.rename({"x": "lat", "y": "lon"})

OBS_SINGLE_NAN = OBS.astype(float).copy()
OBS_SINGLE_NAN.loc[{"x": 1, "y": 1}] = np.nan

OBS_EXTRA_DIM = OBS.expand_dims(time=[1, 2])
OBS_EXTRA_DIM_NAN = OBS_EXTRA_DIM.astype(float).copy()
OBS_EXTRA_DIM_NAN.loc[{"time": 1}] = np.nan

# EXP1_NUM = (((fcst > 2).mean(dim="ens_mem") - (obs>2))**2).sum()
# EXP1_DEN = ((fcst > 2).mean(dim="ens_mem")**2).sum() + ((obs>2)**2).sum()
# EXP1 = 1 - EXP1_NUM / EXP1_DEN
EXP1 = xr.DataArray(1 - 3.25 / 8.25)  # See my calculation above

# fcst_frac = (fcst > 2).mean("ens_mem").rolling(x=2, y=2).mean()
# obs_frac = (obs > 2).rolling(x=2, y=2).mean()
# expected_frac = 1 - ((fcst_frac - obs_frac)**2).sum() / ((fcst_frac**2).sum() + (obs_frac**2).sum())
EXP2 = xr.DataArray(0.69242658)  # See my calculation above

EXP_EXTRA_DIM_NAN = xr.DataArray(EXP2).expand_dims(time=[1, 2]).copy()
EXP_EXTRA_DIM_NAN.loc[{"time": 1}] = np.nan


@pytest.mark.filterwarnings("ignore::scores.utils.DimensionWarning")
@pytest.mark.parametrize(
    (
        "window_size",
        "event_threshold",
        "reduce_dims",
        "preserve_dims",
        "zero_padding",
        "threshold_operator",
    ),
    [
        (
            (2, 2),
            2.0,
            None,
            None,
            None,
            None,
        ),
        (
            (2, 2),
            8.0,
            None,
            None,
            None,
            None,
        ),
        (
            (5, 5),
            2.0,
            None,
            None,
            None,
            None,
        ),
        (
            (5, 5),
            8.0,
            None,
            None,
            None,
            None,
        ),
        (
            (2, 2),
            5.0,
            ["time"],
            None,
            None,
            None,
        ),
        (
            (2, 2),
            5.0,
            None,
            ["time"],
            None,
            None,
        ),
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
            None,
            None,
        ),
        # Test zero_padding argument
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
            True,
            None,
        ),
        # Test threshold_operator argument
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
            None,
            np.less,
        ),
    ],
)
def test_pfss_equals_fss_for_single_member(
    window_size, event_threshold, reduce_dims, preserve_dims, zero_padding, threshold_operator
):
    """
    Test that the pFSS for a single-member ensemble equals the standard FSS.

    These tests inputs were taken from tests/spatial/test_fss.py::test_fss_2d_binary with two more
    tests added at the end to test the zero padding arg and the threshold operator arg.
    """
    fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    obs = sd.continuous_observations(large_size=False)
    fcst_ens = fcst.expand_dims(ens_mem=[1])

    pfss_score = pfss_for_ensemble(
        fcst_ens,
        obs,
        event_threshold=event_threshold,
        window_size=window_size,
        ensemble_member_dim="ens_mem",
        spatial_dims=("lat", "lon"),
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        zero_padding=zero_padding,
        threshold_operator=threshold_operator,
    )
    fss_score = fss_2d(
        fcst,
        obs,
        event_threshold=event_threshold,
        window_size=window_size,
        spatial_dims=("lat", "lon"),
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        zero_padding=zero_padding,
        threshold_operator=threshold_operator,
    )

    xr.testing.assert_equal(pfss_score, fss_score)


@pytest.mark.parametrize(("window_size", "expected"), [((1, 1), EXP1), ((2, 2), EXP2)])
def test_multiple_ens_members(window_size, expected):
    """
    Test that the pFSS for multiple ensemble members returns the correct score
    """
    pfss_score = pfss_for_ensemble(
        FCST,
        OBS,
        event_threshold=2,
        window_size=window_size,
        ensemble_member_dim="ens_mem",
        spatial_dims=("x", "y"),
    )
    xr.testing.assert_allclose(pfss_score, expected)


@pytest.mark.parametrize(
    ("fcst", "obs", "exp"),
    [
        # Perfect skill
        (FCST_PERFECT, OBS, xr.DataArray(1)),
        # Handles divide by zero case
        (FCST * 0, OBS * 0, xr.DataArray(np.nan)),
        # Forecast with no skill
        (FCST, OBS * 0, xr.DataArray(0)),
        # Forecast with no skill (fcsts are non events everywhere)
        (FCST * 0, OBS, xr.DataArray(0)),
    ],
)
def test_edge_cases(fcst, obs, exp):
    """
    Test some edge cases for the pFSS score.
    """
    pfss_score = pfss_for_ensemble(
        fcst,
        obs,
        event_threshold=2,
        window_size=(2, 2),
        ensemble_member_dim="ens_mem",
        spatial_dims=("x", "y"),
    )
    xr.testing.assert_equal(pfss_score, exp)


def test_identical_mems_equal_one_mem():
    """
    Test that identical ensemble members produce the same pFSS as a single-member ensemble.
    """
    fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    obs = sd.continuous_observations(large_size=False)
    fcst_ens1 = fcst.expand_dims(ens_mem=[1])
    fcst_ens2 = fcst.expand_dims(ens_mem=[1, 2])

    pfss_score1 = pfss_for_ensemble(
        fcst_ens1,
        obs,
        event_threshold=0.5,
        window_size=(2, 2),
        ensemble_member_dim="ens_mem",
        spatial_dims=("lat", "lon"),
    )
    pfss_score2 = pfss_for_ensemble(
        fcst_ens2,
        obs,
        event_threshold=0.5,
        window_size=(2, 2),
        ensemble_member_dim="ens_mem",
        spatial_dims=("lat", "lon"),
    )
    xr.testing.assert_equal(pfss_score1, pfss_score2)


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "expected"),
    [
        # Test with a single NaN in the forecast
        # Note that naively using the method to calc `EXP2` above would fail.
        (FCST_SINGLE_NAN, OBS, None, xr.DataArray(0.71387527)),
        # Test with a single Obs NaN in the observation
        (FCST, OBS_SINGLE_NAN, None, xr.DataArray(0.68997669)),
        # Test with all NaNs in the forecast
        (FCST * np.nan, OBS, None, xr.DataArray(np.nan)),
        # Test with all NaNs in the observation
        (FCST, OBS * np.nan, None, xr.DataArray(np.nan)),
        # Test with all NaNs in one ensemble member
        (FCST_MEM_1_NAN, OBS, None, xr.DataArray(0.44875346)),
        # Test with all NaNs at a single point
        (FCST_POINT_ALL_NAN, OBS, None, xr.DataArray(0.68997669)),
        # Test with a single NaN in the extra dimension, preserve the time dimension
        (FCST_EXTRA_DIM_NAN, OBS_EXTRA_DIM, ("time",), EXP_EXTRA_DIM_NAN),
        # Test with a single NaN in the extra dimension, don't preserve the time dimension
        (FCST_EXTRA_DIM_NAN, OBS_EXTRA_DIM_NAN, None, EXP2),
    ],
)
def test_pfss_nan_handling(fcst, obs, preserve_dims, expected):
    """
    Test that the pFSS correctly handles NaN values in the forecast and observation data.

    The expected results were calculated as follows:

        prob = (fcst > 2).where(fcst.notnull()).mean("ens_mem")
        obs_ev = (obs > 2).where(obs.notnull())
        valid = prob.notnull() & obs_ev.notnull()
        fcst_frac = prob.where(valid, 0).rolling(x=2, y=2).mean()
        obs_frac = obs_ev.where(valid, 0).rolling(x=2, y=2).mean()
        exp = 1 - ((fcst_frac - obs_frac)**2).sum() / ((fcst_frac**2).sum() + (obs_frac**2).sum())
    """
    result = pfss_for_ensemble(
        fcst,
        obs,
        event_threshold=2,
        window_size=(2, 2),
        ensemble_member_dim="ens_mem",
        spatial_dims=("x", "y"),
        preserve_dims=preserve_dims,
    )
    xr.testing.assert_allclose(result, expected)


def test_broadcasting():
    """
    Test that the pFSS correctly handles broadcasting of arrays.
    """
    result = pfss_for_ensemble(
        FCST_EXTRA_DIM,
        OBS,
        event_threshold=2,
        window_size=(2, 2),
        ensemble_member_dim="ens_mem",
        spatial_dims=("x", "y"),
        preserve_dims="time",
    )
    xr.testing.assert_allclose(result, xr.DataArray(EXP2).expand_dims(time=[1, 2]))


def test_transposed_arrays():
    """
    Test that the pFSS correctly handles transposed arrays (e.g., dim order changes).
    """
    fcst = FCST.T
    result = pfss_for_ensemble(
        fcst,
        OBS,
        event_threshold=2,
        window_size=(2, 2),
        ensemble_member_dim="ens_mem",
        spatial_dims=("x", "y"),
    )
    xr.testing.assert_allclose(result, xr.DataArray(EXP2))


@pytest.mark.parametrize(
    "fcst,obs, reduce_dims, preserve_dims, threshold_operator, ensemble_member_dim, msg, err",
    [
        # Test ens mem dim not in fcst
        (
            FCST,
            OBS,
            None,
            None,
            None,
            "wrong_dim",
            "`score_specific_fcst_dims` must be a subset of `fcst` dimensions",
            ValueError,
        ),
        # Test ens mem dim in obs
        (FCST, FCST, None, None, None, "ens_mem", "`obs.dims` must not contain any `score_specific_fcst_dims`", ValueError),
        # Test ens mem dim in reduce dims
        (
            FCST,
            OBS,
            ("ens_mem",),
            None,
            None,
            "ens_mem",
            "`reduce_dims` and `preserve_dims` must not contain any `score_specific_fcst_dims`",
            ValueError,
        ),
        # Test ens mem dim in preserve dims
        (
            FCST,
            OBS,
            None,
            ("ens_mem",),
            None,
            "ens_mem",
            "`reduce_dims` and `preserve_dims` must not contain any `score_specific_fcst_dims`",
            ValueError,
        ),
        # Test threshold_operator not supported
        (FCST, OBS, None, None, np.equal, "ens_mem", "Invalid operator specified. Allowed operators:", ValueError),
        # Test misaligned coordinates
        (FCST, OBS, None, None, None, "ens_mem", "invalid window size,", DimensionError),
        # Test spatial dims not in fcst
        (FCST_RENAMED_DIM, OBS, None, None, None, "ens_mem", "not found in forecast dimensions.", DimensionError),
        # Test spatial dims not in obs
        (FCST, OBS_RENAMED_DIM, None, None, None, "ens_mem", "not found in observation dimensions.", DimensionError),
    ],
)
def test_pfss_raises(fcst, obs, reduce_dims, preserve_dims, threshold_operator, ensemble_member_dim, msg, err):
    """
    Test that the pFSS raises appropriate errors for invalid inputs.
    """
    window_size = (2, 2)
    if err is DimensionError:
        window_size = (10, 10)

    with pytest.raises(err, match=msg):
        pfss_for_ensemble(
            fcst,
            obs,
            event_threshold=2,
            window_size=window_size,
            ensemble_member_dim=ensemble_member_dim,
            spatial_dims=("x", "y"),
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
            threshold_operator=threshold_operator,
        )


def test_pfss_from_spatial():
    """
    Test that we can also import pfss from scores.spatial.
    """
    from scores.spatial import pfss_for_ensemble as spatial_pfss # pylint: disable=import-outside-toplevel
    assert pfss_for_ensemble is spatial_pfss
    assert "pfss_for_ensemble" in scores.spatial.__all__

    with pytest.raises(AttributeError, match="has no attribute 'fake_function'"):
        getattr(scores.spatial, "fake_function")