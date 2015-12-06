from pyspark.mllib.classification import MLNaiveBayesModel
from pyspark.mllib.linalg import _convert_to_vector
from pyspark.mllib.linalg import Vectors
from pyspark import RDD
import numpy as np
import math


def train_model(data: RDD, l=1.0) -> MLNaiveBayesModel:
    aggregated = data.flatMap(lambda x:
                              [(l, x['features']) for l in x['labels']]) \
        .combineByKey(lambda v: (1, v),
                      lambda c, v: (c[0] + 1, c[1] + v),
                      lambda c1, c2: (c1[0] + c2[0], c1[1] + c2[1])) \
        .sortBy(lambda x: x[0]) \
        .collect()
    num_labels = len(aggregated)
    num_documents = data.count()
    num_features = aggregated[0][1][1].size
    labels = np.zeros(num_labels)
    pi = np.zeros(num_labels, dtype=int)
    theta = np.zeros((num_labels, num_features))
    pi_log_denom = math.log(num_documents + num_labels * l)
    i = 0
    for (label, (n, sum_term_freq)) in aggregated:
        labels[i] = label
        pi[i] = math.log(n + l) - pi_log_denom

        sum_term_freq_dense = sum_term_freq.toarray()
        theta_log_denom = math.log(sum_term_freq.sum() + num_features * l)
        theta[i, :] = np.log(sum_term_freq_dense + l) - theta_log_denom
        i += 1
    return MLNaiveBayesModel(labels, pi, theta)
