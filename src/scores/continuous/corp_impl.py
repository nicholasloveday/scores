"""
Implementation of the corp decomposition
"""
from scores.continuous import (
    mse,
    mae,
    quantile_score,
    tw_absolute_error,
    tw_quantile_score,
    tw_squared_error,
    consistent_quantile_score,
)
from scores.probability import brier_score
from scores.categorical import firm
from scores.processing import broadcast_and_match_nan
from scores.processing.isoreg_impl import isotonic_fit

import numpy as np
import xarray as xr

MEAN_SCORES = [mse, tw_squared_error, brier_score]
QUANTILE_SCORES = [quantile_score, consistent_quantile_score, tw_quantile_score]
MEDIAN_SCORES = [mae, tw_absolute_error]
FLEXIBLE_SCORES = [firm]


def corp_decomposition(fcst, obs, scoring_function, score_kwargs):
    """
    Calculates the CORP decomposition for a given scoring function.
    """
    # TODO: insert logic to determine functional
    if scoring_function in MEAN_SCORES:
        functional = "mean"
    elif scoring_function in QUANTILE_SCORES or MEDIAN_SCORES:
        functional = "quantile"
    # elif scoring_function in FLEXIBLE_SCORES:

    ref_forecast, _ = broadcast_and_match_nan(obs.mean(), fcst)

    # TODO: If functional is mean add some logic that just uses scipy's implementation
    iso_dict = isotonic_fit(
        fcst=fcst,
        obs=obs,
        functional=functional,
    )
    recalibrated_fcst = xr.full_like(other=fcst, fill_value=np.nan)
    recalibrated_fcst.values = iso_dict["regression_func"](fcst)

    score = scoring_function(fcst, obs, score_kwargs)
    recal_score = scoring_function(recalibrated_fcst, obs, score_kwargs)
    clim_score = scoring_function(ref_forecast, obs, score_kwargs)

    # TODO: Add in logic here to handle the dataset that FIRM returns
    mcb = score - recal_score
    dsc = clim_score - recal_score
    unc = clim_score

    return score, mcb, dsc, unc
