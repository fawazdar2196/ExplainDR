import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(probs, t):
    # log probabilities equal the logits up to a constant, so dividing by t rescales confidence
    logits = np.log(np.clip(probs, 1e-7, 1.0)) / t
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs, labels):
    # one number, chosen to make the probabilities fit the true grades best (lowest log loss)
    def nll(t):
        p = apply_temperature(probs, t)
        return -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def calibration_bins(confidence, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            rows.append((float(confidence[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def expected_calibration_error(confidence, correct, n_bins=10):
    # average gap between confidence and accuracy, weighted by how many photos fall in each bin
    n = len(confidence)
    return float(sum(count / n * abs(acc - conf) for conf, acc, count in calibration_bins(confidence, correct, n_bins)))
