from pyspark import RDD
import numpy as np


class Metric:
    def __init__(self, name: str, verbose=False):
        self._name = name
        self._results = []
        self._verbose = verbose

    @property
    def name(self) -> str:
        return self._name

    @property
    def results(self) -> list:
        return self._results

    @property
    def avg(self) -> float:
        return np.average(self._results)

    def evaluate(self, labels_and_predictions: RDD) -> float:
        pass


class AccuracyMetric(Metric):
    def __init__(self, pred_n: int, intersect_n: int):
        self._pred_n = pred_n
        self._intersect_n = intersect_n
        super(AccuracyMetric, self).__init__(name='Accuracy', verbose=False)

    def evaluate(self, labels_and_predictions: RDD) -> float:
        tp = labels_and_predictions \
            .map(lambda x:
                 (set(x[0]),
                  set(features for features, weights in x[1][:self._pred_n]))) \
            .filter(lambda x:
                    len(x[0].intersection(x[1])) >= self._intersect_n)
        accuracy = 100.0 * tp.count() / labels_and_predictions.count()
        if self._verbose:
            print('accuracy: ', accuracy)
        self._results.append(accuracy)
        return accuracy
