import numpy as np
import pytest

from utils.metrics import ClassificationMetrics, SegmentationMetrics


def test_dice_identical_and_disjoint(synthetic_mask):
    m = SegmentationMetrics()
    assert m.dice_coefficient(synthetic_mask, synthetic_mask) == pytest.approx(1.0)
    other = np.zeros_like(synthetic_mask)
    other[0:5, 0:5] = 1
    assert m.dice_coefficient(synthetic_mask, other) < 1e-3


def test_dice_known_overlap():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 0])
    # 2*1 / (2+1)
    assert SegmentationMetrics().dice_coefficient(a, b, smooth=0) == pytest.approx(2 / 3)


def test_iou_known_overlap():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 0])
    assert SegmentationMetrics().iou_score(a, b, smooth=0) == pytest.approx(0.5)


def test_hausdorff_is_inf_for_empty_prediction(synthetic_mask):
    assert SegmentationMetrics().hausdorff_distance(synthetic_mask, np.zeros_like(synthetic_mask)) == float("inf")


def test_classification_metrics_perfect_and_chance():
    cm = ClassificationMetrics(num_classes=3)
    y = np.array([0, 1, 2, 0, 1, 2])
    assert cm.calculate_metrics(y, y)["accuracy"] == 1.0
    assert cm.calculate_metrics(y, np.zeros_like(y))["accuracy"] == pytest.approx(1 / 3)
