import numpy as np
import pytest

from utils.preprocessing import MammographyPreprocessor


@pytest.fixture
def pre():
    return MammographyPreprocessor()


def test_preprocess_output_shape_and_range(pre, synthetic_scan):
    out = pre.preprocess_image(synthetic_scan, target_size=(64, 64))
    assert out.shape == (64, 64)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_resize_keeps_aspect_ratio_and_pads_to_target(pre):
    wide = np.full((40, 100), 200, np.uint8)
    out = pre.resize_image(wide, (64, 64))
    assert out.shape == (64, 64)
    assert out[0].max() == 0          # padded rows are black
    assert out[32].max() > 0          # image content in the middle


def test_normalize_constant_image_does_not_divide_by_zero(pre):
    out = pre.normalize_intensity(np.full((8, 8), 7, np.uint8))
    assert np.isfinite(out).all()


def test_remove_dicom_tags_blanks_small_bright_text(pre):
    img = np.full((64, 64), 100, np.uint8)
    img[5:8, 5:20] = 250               # a small bright "label"
    assert pre.remove_dicom_tags(img)[5:8, 5:20].max() == 0


def test_unknown_methods_raise(pre):
    with pytest.raises(ValueError):
        pre.noise_removal(np.zeros((8, 8), np.uint8), method="nope")
    with pytest.raises(ValueError):
        pre.enhance_contrast(np.zeros((8, 8), np.uint8), method="nope")


@pytest.mark.xfail(strict=True, reason="BUG (review.md): remove_dicom_tags erases every bright "
                   "blob under 5000 px, so a small bright lesion/calcification is deleted as 'text'")
def test_small_bright_lesion_is_preserved(pre):
    img = np.full((64, 64), 100, np.uint8)
    img[30:34, 30:34] = 250
    assert pre.remove_dicom_tags(img)[30:34, 30:34].max() > 0


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="BUG (review.md): remove_artifacts calls np.percentile on an empty "
                   "selection when the image is all black")
def test_blank_image_does_not_crash(pre):
    pre.preprocess_image(np.zeros((64, 64), np.uint8))
