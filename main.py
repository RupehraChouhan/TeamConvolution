from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import timeit
import exponential
from collections import OrderedDict
from pprint import pformat


def run(algorithm, x_test, y_test, algorithm_name='Algorithm'):
    print('Running {}...'.format(algorithm_name))
    start = timeit.default_timer()
    np.random.seed(0)
    predicted_y_test = algorithm.run(x_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    correct_predict = (y_test == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    print("Result for {}: ".format(algorithm_name))
    print("Correct Predict: {}/{} total \tAccuracy: {:5f} \tTime: {:2f}".format(correct_predict, len(y_test), accuracy,
                                                                                run_time))
    return correct_predict, accuracy, run_time


if __name__ == '__main__':
    # Read MNIST dataset
    mnist = read_data_sets("data", one_hot=False)
    result = [
        OrderedDict(
            first_name='Insert your First name here',
            last_name='Insert your Last name here',
        )
    ]
    for algorithm in [exponential]:
        x_valid, y_valid = mnist.validation._images, mnist.validation.labels
        if algorithm.__name__ == 'knn':
            x_valid, y_valid = x_valid[:1000], y_valid[:1000]
        correct_predict, accuracy, run_time = run(algorithm, x_valid, y_valid, algorithm_name=algorithm.__name__)
        result.append(OrderedDict(
            algorithm_name=algorithm.__name__,
            correct_predict=correct_predict,
            accuracy=accuracy,
            run_time=run_time
        ))
        with open('result.txt', 'w') as f:
            f.writelines(pformat(result, indent=4))
