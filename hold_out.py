from pyspark import RDD


def shuffle_and_split(data: RDD, fold_n: int, seed: int = 0) -> list:
    fold_weights = [1 / fold_n] * fold_n
    return data.randomSplit(fold_weights, seed)


def hold_out(sc, data: RDD, k: int, model_builder, metrics: list) -> list:
    folds = shuffle_and_split(data, k)
    for i in range(k):
        test = folds[i]
        training = sc.union(folds[:i] + folds[i + 1:])
        model = model_builder(training)
        model_br = sc.broadcast(model)
        labels_and_predictions = test \
            .map(lambda x: (x['labels'], model_br.value.predict_all(x['features'])))
        for metric in metrics:
            metric.evaluate(labels_and_predictions)
    return metrics
