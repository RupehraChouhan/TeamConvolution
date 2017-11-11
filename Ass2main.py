import numpy as np
import timeit
from code.mnisdd_classify import mnisdd_classify
from collections import OrderedDict
from pprint import pformat

if __name__ == '__main__':
    TRAIN = True
    X_train = np.load('./MNISTDD_train+valid/train_X.npy')
    if TRAIN:
        mnisdd_classify(X_train)
    X_test = np.load('./MNISTDD_train+valid/valid_X.npy')
    y_test = np.load('./MNISTDD_train+valid/valid_Y.npy')
    bboxes  = np.load('./MNISTDD_train+valid/train_bboxes.npy')

    start = timeit.default_timer()
    np.random.seed(0)
    predicted_test_labels = test(X_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start
    correct_predict = (y_test.flatten() == predicted_test_labels.flatten()).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)
    result = OrderedDict(
        correct_predict=correct_predict,
        accuracy=accuracy,
        run_time=run_time
    )
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))
