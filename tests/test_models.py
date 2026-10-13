import numpy as np
import pytest
import tensorflow as tf

from models.efficientnet import EfficientNetClassifier
from models.unet import UNet


def test_unet_output_shape_and_range():
    unet = UNet(input_shape=(32, 32, 1), filters=4)
    out = unet.model.predict(np.random.rand(2, 32, 32, 1).astype("float32"), verbose=0)
    assert out.shape == (2, 32, 32, 1)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_unet_dice_loss_zero_for_identical_masks():
    unet = UNet(input_shape=(16, 16, 1), filters=4)
    m = tf.constant((np.random.rand(1, 16, 16, 1) > 0.5).astype("float32"))
    assert float(unet.dice_loss(m, m)) == pytest.approx(0.0, abs=1e-5)


def test_unet_predict_returns_binary_mask():
    unet = UNet(input_shape=(16, 16, 1), filters=4)
    mask = unet.predict(np.random.rand(16, 16).astype("float32"))
    assert mask.shape == (16, 16) and set(np.unique(mask)) <= {0, 1}


def test_efficientnet_builds_without_download_and_outputs_probabilities():
    clf = EfficientNetClassifier(num_classes=3, input_shape=(32, 32, 3), weights=None)
    out = clf.model.predict(np.random.rand(2, 32, 32, 3).astype("float32"), verbose=0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.xfail(strict=True, reason="BUG (review.md): fine_tune() unfreezes model.layers[3], "
                   "which is GlobalAveragePooling2D; the EfficientNet base is layers[2], so it never unfreezes")
def test_fine_tune_targets_the_efficientnet_base():
    clf = EfficientNetClassifier(num_classes=3, input_shape=(32, 32, 3), weights=None)
    assert isinstance(clf.model.layers[3], tf.keras.Model)
