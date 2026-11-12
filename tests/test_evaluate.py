import json

import numpy as np

from absa.evaluate import confusion_by_aspect, evaluate

C, B = 1, 3  # cột CAMERA, BATTERY


def make(rows):
    Y = np.zeros((len(rows), 10), dtype=np.int8)
    for i, r in enumerate(rows):
        for col, v in r.items():
            Y[i, col] = v
    return Y


def test_perfect():
    Y = make([{C: 1}, {B: 3}, {}])
    m = evaluate(Y, Y)
    assert m["aspect_macro_f1"] == 1.0 and m["sentiment_macro_f1"] == 1.0
    assert m["exact_match"] == 1.0 and m["polarity_macro_f1_detected"] == 1.0


def test_right_aspect_wrong_polarity():
    m = evaluate(make([{C: 1}]), make([{C: 3}]))
    assert m["aspect_macro_f1"] == 1.0
    assert m["sentiment_macro_f1"] == 0.0
    assert m["exact_match"] == 0.0


def test_all_none_rows_and_empty_prediction_no_crash():
    m = evaluate(make([{}, {}]), make([{}, {C: 1}]))
    assert m["exact_match"] == 0.5
    assert m["aspect_macro_f1"] == 0.0 and m["sentiment_macro_f1"] == 0.0
    assert m["polarity_macro_f1_detected"] == 0.0


def test_macro_ignores_absent_aspect():
    m = evaluate(make([{C: 1}, {B: 1}]), make([{C: 1}, {}]))
    assert m["aspect_macro_f1"] == 0.5
    assert m["per_aspect"]["STORAGE"]["support"] == 0


def test_polarity_detected():
    m = evaluate(make([{C: 1}, {B: 3}]), make([{C: 1}, {B: 1}]))
    assert abs(m["polarity_macro_f1_detected"] - 1 / 3) < 1e-9  # Positive F1=2/3, Negative F1=0


def test_json_serializable():
    json.dumps(evaluate(make([{C: 1}]), make([{C: 1}])))


def test_confusion_by_aspect():
    cm = confusion_by_aspect(make([{C: 1}, {C: 3}, {}]), make([{C: 1}, {C: 1}, {C: 2}]), "CAMERA")
    assert cm.shape == (4, 4) and cm.sum() == 3
    assert cm[1, 1] == 1 and cm[3, 1] == 1 and cm[0, 2] == 1
