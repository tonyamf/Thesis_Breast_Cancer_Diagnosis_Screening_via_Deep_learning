"""Smoke test on tiny synthetic data. Not a result: no real data, no training to convergence.

Run from this folder:  python smoke_test.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from utils.preprocessing import MammographyPreprocessor
from utils.augmentation import MammographyAugmentation
from utils.metrics import ClassificationMetrics, SegmentationMetrics
from models.unet import UNet
from models.efficientnet import EfficientNetClassifier
from models.meta_pseudo_labels import MetaPseudoLabels

rng = np.random.default_rng(0)
tf.random.set_seed(0)

# 1. preprocessing + augmentation on a synthetic 128x128 grey image with a bright blob
img = (rng.random((128, 128)) * 80).astype(np.uint8)
img[40:80, 40:80] += 120
mask = np.zeros((128, 128), np.uint8)
mask[40:80, 40:80] = 1
pre = MammographyPreprocessor().preprocess_image(img)
print("preprocess ->", np.asarray(pre).shape)
aug_img, aug_mask = MammographyAugmentation().apply_albumentations(img, mask)
print("augment ->", aug_img.shape, aug_mask.shape)

# 2. U-Net: build, one training step
x = rng.random((4, 64, 64, 1)).astype("float32")
y = (rng.random((4, 64, 64, 1)) > 0.5).astype("float32")
unet = UNet(input_shape=(64, 64, 1), filters=8)
unet.compile_model()
h = unet.model.fit(x, y, epochs=1, batch_size=2, verbose=0)
print("unet 1 epoch loss:", float(h.history["loss"][0]))

# 3. EfficientNet-B0 classifier (random init: no download), one training step
xc = rng.random((6, 64, 64, 3)).astype("float32")
yc = tf.one_hot(rng.integers(0, 3, 6), 3).numpy()
clf = EfficientNetClassifier(num_classes=3, input_shape=(64, 64, 3), weights=None, freeze_base=False)
clf.compile_model()
h = clf.model.fit(xc, yc, epochs=1, batch_size=3, verbose=0)
print("classifier 1 epoch loss:", float(h.history["loss"][0]))

# 4. Meta Pseudo Labels: one epoch with tiny teacher/student
def tiny():
    return tf.keras.Sequential([tf.keras.layers.Input((64, 64, 3)), tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(3, activation="softmax")])
mpl = MetaPseudoLabels(tiny(), tiny(), num_classes=3, confidence_threshold=0.3)
hist = mpl.train(xc, np.argmax(yc, 1), rng.random((6, 64, 64, 3)).astype("float32"),
                 xc, np.argmax(yc, 1), epochs=1, batch_size=3)
print("mpl history keys:", list(hist))

# 5. metrics
yt = rng.integers(0, 3, 30)
print("cls metrics keys:", list(ClassificationMetrics(num_classes=3).calculate_metrics(yt, yt).keys())[:4])
print("dice(identical masks):", SegmentationMetrics().dice_coefficient(mask, mask))
print("SMOKE TEST PASSED")
