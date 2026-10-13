"""Make python_files/ importable and keep TensorFlow quiet and deterministic."""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_files"))

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    tf.random.set_seed(0)


@pytest.fixture
def synthetic_scan():
    """128x128 uint8 'mammogram': dim noisy background with one large bright blob."""
    rng = np.random.default_rng(0)
    img = (rng.random((128, 128)) * 80).astype(np.uint8)
    img[40:80, 40:80] += 120
    return img


@pytest.fixture
def synthetic_mask():
    mask = np.zeros((128, 128), np.uint8)
    mask[40:80, 40:80] = 1
    return mask
