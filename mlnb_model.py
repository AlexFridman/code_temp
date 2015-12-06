import numpy as np


class MLNBModel:
    def __init__(self, labels: np.ndarray, pi: np.ndarray, theta: np.ndarray):
        self.labels = labels
        self.pi = pi
        self.theta = theta

    @staticmethod
    def scale(x: np.ndarray) -> np.ndarray:
        max_x = x.max()
        min_x = x.min()
        return (x - min_x) / (max_x - min_x)

    def predict(self, x: np.ndarray, labels_n: int = None) -> list:
        log_probs = self.pi + x.dot(self.theta.transpose())
        scaled_log_probs = self.scale(log_probs)
        ordered_idxs = scaled_log_probs.argsort()[::-1][:labels_n]
        labels_and_log_probs = list(zip(self.labels[ordered_idxs],
                                        scaled_log_probs[ordered_idxs]))
        return labels_and_log_probs
