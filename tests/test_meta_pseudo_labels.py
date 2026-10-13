import numpy as np
import pytest
import tensorflow as tf

from models.meta_pseudo_labels import MetaPseudoLabels


def tiny(n=3):
    return tf.keras.Sequential([tf.keras.layers.Input((4,)),
                                tf.keras.layers.Dense(n, activation="softmax")])


def weights(model):
    return [w.numpy().copy() for w in model.trainable_variables]


def changed(before, after):
    return any(not np.allclose(b, a) for b, a in zip(before, after))


@pytest.fixture
def data():
    x = np.random.rand(8, 4).astype("float32")
    y = tf.one_hot(np.random.randint(0, 3, 8), 3).numpy()
    return x, y


def confident_teacher():
    t = tiny()
    t.layers[0].set_weights([np.array([[50, -50, -50]] * 4, "float32"), np.zeros(3, "float32")])
    return t


def test_temperature_scaling_gives_a_distribution():
    m = MetaPseudoLabels(tiny(), tiny())
    p = m.temperature_scaling(tf.constant([[1.0, 2.0, 3.0]]), 1.0).numpy()
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2


def test_higher_temperature_flattens_the_distribution():
    m = MetaPseudoLabels(tiny(), tiny())
    logits = tf.constant([[1.0, 2.0, 3.0]])
    assert m.temperature_scaling(logits, 5.0).numpy().max() < m.temperature_scaling(logits, 1.0).numpy().max()


def test_pseudo_labels_are_one_hot_and_threshold_zero_keeps_everything(data):
    x, _ = data
    m = MetaPseudoLabels(tiny(), tiny(), confidence_threshold=0.0)
    labels, mask = m.generate_pseudo_labels(x)
    assert labels.shape == (8, 3)
    np.testing.assert_allclose(labels.numpy().sum(axis=1), 1.0)
    assert bool(tf.reduce_all(mask))


def test_student_step_updates_the_student(data):
    x, y = data
    m = MetaPseudoLabels(tiny(), tiny(), confidence_threshold=0.0)
    labels, mask = m.generate_pseudo_labels(x)
    before = weights(m.student)
    m.student_training_step(x, y, x, labels, mask)
    assert changed(before, weights(m.student))


def test_teacher_step_restores_student_weights(data):
    x, y = data
    m = MetaPseudoLabels(tiny(), tiny(), confidence_threshold=0.0)
    before = weights(m.student)
    m.teacher_training_step(x, y, x, before)
    for b, a in zip(before, weights(m.student)):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_train_runs_and_reports_history(data):
    x, y = data
    m = MetaPseudoLabels(tiny(), tiny(), confidence_threshold=0.0)
    hist = m.train(x, y.argmax(1), x, x, y.argmax(1), epochs=1, batch_size=4)
    assert set(hist) == {"student_loss", "teacher_loss", "pseudo_label_ratio", "validation_accuracy"}
    assert len(hist["student_loss"]) == 1


@pytest.mark.xfail(strict=True, reason="BUG (review.md): the teacher gets no gradient (hard argmax "
                   "labels + assign()), so the 'meta' update never trains the teacher")
def test_teacher_weights_change_after_a_meta_step(data):
    x, y = data
    m = MetaPseudoLabels(tiny(), tiny(), confidence_threshold=0.0)
    before = weights(m.teacher)
    m.teacher_training_step(x, y, x, weights(m.student))
    assert changed(before, weights(m.teacher))


@pytest.mark.xfail(strict=True, reason="BUG (review.md): softmax is applied to a model that already "
                   "outputs softmax, so with 3 classes max confidence is e/(e+2)=0.576 < 0.8 default; "
                   "no pseudo label ever passes the threshold")
def test_a_certain_teacher_passes_the_default_threshold(data):
    x, _ = data
    m = MetaPseudoLabels(confident_teacher(), tiny())      # default confidence_threshold=0.8
    assert float(tf.reduce_mean(tf.cast(m.generate_pseudo_labels(x)[1], tf.float32))) == 1.0
