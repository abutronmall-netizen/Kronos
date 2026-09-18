"""Equivalence tests for the cached inference path.

`cached_auto_regressive_inference` is an optimization, so the only thing that
makes it safe is that it produces exactly what the reference implementation
produces. These tests pin that claim, and pin the boundary where the
optimization declines to run rather than returning near-enough numbers.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"

FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CTX_LEN = 512
PRED_LEN = 5
# Sized so the whole generation fits inside max_context, which is the regime the
# cached path supports; one bar more would trigger eviction.
CTX_LEN = MAX_CTX_LEN - PRED_LEN
SEED = 123
DEVICE = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(scope="module")
def predictor():
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    return KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)


@pytest.fixture(scope="module")
def series():
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    context = df.iloc[:CTX_LEN]
    future = df["timestamps"].iloc[CTX_LEN:CTX_LEN + PRED_LEN]
    return (
        context[FEATURE_NAMES].reset_index(drop=True),
        context["timestamps"].reset_index(drop=True),
        future.reset_index(drop=True),
    )


def _predict(predictor, series, sample_count, use_cache, return_paths=False):
    features, x_timestamp, y_timestamp = series
    x, x_stamp, y_stamp, x_mean, x_std = predictor._prepare_series(features, x_timestamp, y_timestamp)
    set_seed(SEED)
    with torch.no_grad():
        out = predictor.generate(
            x, x_stamp, y_stamp, PRED_LEN, 1.0, 0, 0.9, sample_count, False,
            return_paths=return_paths, use_cache=use_cache,
        )
    return out * (x_std + 1e-5) + x_mean


@pytest.mark.parametrize("sample_count", [1, 4])
def test_cached_matches_reference(predictor, series, sample_count):
    """Cached and reference paths agree exactly when no eviction is needed."""
    cached = _predict(predictor, series, sample_count, use_cache=True)
    reference = _predict(predictor, series, sample_count, use_cache=False)
    np.testing.assert_array_equal(cached, reference)


def test_paths_are_not_pre_averaged(predictor, series):
    """`return_paths` exposes the distribution, and its mean is the default output."""
    sample_count = 4
    paths = _predict(predictor, series, sample_count, use_cache=True, return_paths=True)
    averaged = _predict(predictor, series, sample_count, use_cache=True)

    assert paths.shape == (1, sample_count, PRED_LEN, len(FEATURE_NAMES))
    np.testing.assert_allclose(paths.mean(axis=1), averaged, rtol=1e-5)
    # A distribution that collapsed to its mean would carry no usable dispersion.
    assert paths[0, :, -1, FEATURE_NAMES.index("close")].std() > 0


def test_cached_path_refuses_eviction(predictor):
    """The cached implementation rejects a context it cannot serve exactly."""
    from model.kronos import cached_auto_regressive_inference

    x = torch.zeros(1, MAX_CTX_LEN, len(FEATURE_NAMES))
    stamp = torch.zeros(1, MAX_CTX_LEN, 5)
    with pytest.raises(ValueError, match="max_context"):
        cached_auto_regressive_inference(
            predictor.tokenizer, predictor.model, x, stamp, stamp[:, :PRED_LEN],
            MAX_CTX_LEN, PRED_LEN,
        )


def test_overflow_falls_back_to_reference(predictor):
    """An overflowing context warns and still returns reference results."""
    from model.kronos import auto_regressive_inference

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    context = df.iloc[:MAX_CTX_LEN]
    future = df["timestamps"].iloc[MAX_CTX_LEN:MAX_CTX_LEN + PRED_LEN].reset_index(drop=True)
    x, x_stamp, y_stamp, _, _ = predictor._prepare_series(
        context[FEATURE_NAMES].reset_index(drop=True),
        context["timestamps"].reset_index(drop=True),
        future,
    )
    args = (predictor.tokenizer, predictor.model,
            torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp),
            MAX_CTX_LEN, PRED_LEN)

    set_seed(SEED)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="falling back"):
        fell_back = auto_regressive_inference(*args, use_cache=True)
    set_seed(SEED)
    with torch.no_grad():
        reference = auto_regressive_inference(*args, use_cache=False)

    np.testing.assert_array_equal(fell_back, reference)
