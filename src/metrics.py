import math

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

LABELS = [0, 1, 2, 3, 4]


def grading_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        # kappa is undefined when every prediction is the same grade
        "qwk": 0.0 if math.isnan(qwk) else qwk,
    }
