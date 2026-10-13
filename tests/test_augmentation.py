import numpy as np
import pytest

from utils.augmentation import MammographyAugmentation


def test_augment_dataset_grows_and_keeps_labels(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda *a, **k: 0.99)   # skip the random medical branches
    aug = MammographyAugmentation()
    imgs = np.random.rand(4, 32, 32, 1).astype("float32")
    out = aug.augment_dataset(imgs, np.array([0, 1, 2, 0]), augmentation_factor=2)
    assert out["images"].shape == (8, 32, 32, 1)
    assert out["labels"].tolist() == [0, 1, 2, 0, 0, 1, 2, 0]


def test_image_and_mask_get_same_geometry(synthetic_scan, synthetic_mask):
    aug = MammographyAugmentation()
    img, mask = aug.apply_albumentations(synthetic_scan, synthetic_mask)
    assert img.shape[:2] == mask.shape == (128, 128)


@pytest.mark.xfail(strict=True, reason="BUG (review.md): augmented copies come back as 0-255 "
                   "while the originals are 0-1, so one array mixes two intensity scales")
def test_augmented_images_keep_the_input_intensity_scale(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda *a, **k: 0.99)
    aug = MammographyAugmentation()
    imgs = np.random.rand(4, 32, 32, 1).astype("float32")
    out = aug.augment_dataset(imgs, np.array([0, 1, 2, 0]), augmentation_factor=2)["images"]
    assert out.max() <= 1.0 + 1e-6


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="BUG (review.md): the JPEG branch of medical_specific_augmentation decodes "
                   "as grayscale, so an RGB image comes back 2-D and np.array() of the batch fails "
                   "(10% chance per image)")
def test_augment_dataset_survives_the_jpeg_branch_on_rgb_images(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda *a, **k: 0.0)    # take every medical branch
    aug = MammographyAugmentation()
    imgs = np.random.rand(4, 32, 32, 3).astype("float32")
    aug.augment_dataset(imgs, np.array([0, 1, 2, 0]), augmentation_factor=2)
