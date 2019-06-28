import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from BoostedRegressionForest import RandomForest
from BoostedRegressionForest import mae
from BoostedRegressionForest import mse


# tests dependency between changing sample size and counted RMSE and accuracy
def test_sample_size(x_train, x_test, y_train, y_test, boosted):
    print("Test sample size...boosted = ", boosted)
    rmse_errors = []
    accuracy = []
    for i in range(1, int(len(x_train)), int(len(x_train) / 10)):

        y_predicted = RandomForest(x_train, y_train, 30, i, boosted=boosted).predict(x_test)
        results = zip(y_predicted, y_test)

        right = 0
        for y1, y2 in results:
            # print(y1, " : ", y2, round(y1) == y2)
            if round(y1) == y2:
                right += 1
        rmse_errors.append(np.sqrt(mse(y_test, y_predicted)))
        accuracy.append(right / len(y_predicted) * 100)

    print(rmse_errors)
    print(accuracy)
    plt.figure(1)
    plt.plot(np.array(range(1, int(len(x_train)), int(len(x_train) / 10))), rmse_errors)
    plt.title('Root Mean Squared Error, number of trees=30, min in leaf=1')
    plt.ylabel('RMSE')
    plt.xlabel('Sample size')
    plt.figure(2)
    plt.plot(np.array(range(1, int(len(x_train)), int(len(x_train) / 10))), accuracy)
    plt.title('Accuracy, number of trees=30, min in leaf=1')
    plt.ylabel('Accuracy')
    plt.xlabel('Sample size')
    plt.show()


# tests dependency between changing number of trees in forest and counted RMSE and accuracy
def test_number_of_trees(x_train, x_test, y_train, y_test, boosted):
    print("Test number of trees start...boosted = ", boosted)
    rmse_errors = []
    accuracy = []
    for i in range(1, 142, 10):

        y_predicted = RandomForest(x_train, y_train, i, int(len(x_train) / 3), boosted=boosted).predict(x_test)
        results = zip(y_predicted, y_test)

        right = 0
        for y1, y2 in results:
            # print(y1, " : ", y2, round(y1) == y2)
            if round(y1) == y2:
                right += 1
        rmse_errors.append(np.sqrt(mse(y_test, y_predicted)))
        accuracy.append(right / len(y_predicted) * 100)

    print(rmse_errors)
    print(accuracy)
    plt.figure(3)
    plt.plot(np.array(range(1, 142, 10)), rmse_errors)
    plt.title('Root Mean Squared Error, sample size = len(x_train)/3, min in leaf=1')
    plt.ylabel('RMSE')
    plt.xlabel('number of trees')
    plt.figure(4)
    plt.plot(np.array(range(1, 142, 10)), accuracy)
    plt.title('Accuracy, sample size = len(x_train)/3, min in leaf=1')
    plt.ylabel('Accuracy')
    plt.xlabel('number of trees')
    plt.show()


# tests dependency between changing min number of observations in leaf and counted RMSE and accuracy
def test_min_in_leaf(x_train, x_test, y_train, y_test, boosted):
    print("Test min in leaf start...boosted = ", boosted)
    rmse_errors = []
    accuracy = []
    for i in range(1, 42, 5):

        y_predicted = RandomForest(x_train, y_train, 30, int(len(x_train) / 3), min_in_leaf=i,
                                   boosted=boosted).predict(x_test)
        results = zip(y_predicted, y_test)

        right = 0
        for y1, y2 in results:
            # print(y1, " : ", y2, round(y1) == y2)
            if round(y1) == y2:
                right += 1
        rmse_errors.append(np.sqrt(mse(y_test, y_predicted)))
        accuracy.append(right / len(y_predicted) * 100)

    print(rmse_errors)
    print(accuracy)
    plt.figure(5)
    plt.plot(np.array(range(1, 42, 5)), rmse_errors)
    plt.title('Root Mean Squared Error, sample size = len(x_train)/3, number of trees=30')
    plt.ylabel('RMSE')
    plt.xlabel('min_in_leaf')
    plt.figure(6)
    plt.plot(np.array(range(1, 42, 5)), accuracy)
    plt.title('Accuracy, sample size = len(x_train)/3, number of trees=30')
    plt.ylabel('Accuracy')
    plt.xlabel('min_in_leaf')
    plt.show()


# calls tests for both standard forest and boosted one
def test_parameteres(entry_dataset):
    dataset = entry_dataset.iloc[np.random.permutation(entry_dataset.shape[0]), :]

    x_dataset = dataset.iloc[:, 0:11].values
    y_dataset = dataset.iloc[:, 11].values

    x_train, x_test = np.split(x_dataset, [-(int)(len(x_dataset) * 0.3)], axis=0)
    y_train, y_test = np.split(y_dataset, [-(int)(len(y_dataset) * 0.3)], axis=0)

    print("Standard forest")
    start_time = time.time()
    test_sample_size(x_train, x_test, y_train, y_test, False)
    test_number_of_trees(x_train, x_test, y_train, y_test, False)
    test_min_in_leaf(x_train, x_test, y_train, y_test, False)
    print("Training time not boosted: ", time.time() - start_time)

    start_time = time.time()
    print("Boosted forest")
    test_sample_size(x_train, x_test, y_train, y_test, True)
    test_number_of_trees(x_train, x_test, y_train, y_test, True)
    test_min_in_leaf(x_train, x_test, y_train, y_test, True)
    print("Training time boosted: ", time.time() - start_time)


# counts in 10 tries averaged mse, rmse, mae and accuracy for boosted and not boosted forests with default parameters
# and different test:train splitting ratio
def test_default_values(entry_dataset, boosted, splitting_ratio):
    start_time = time.time()
    for i in range(10):
        rmse_errors = []
        mse_errors = []
        mae_errors = []
        accuracy = []
        dataset = entry_dataset.iloc[np.random.permutation(entry_dataset.shape[0]), :]

        x_dataset = dataset.iloc[:, 0:11].values
        y_dataset = dataset.iloc[:, 11].values

        x_train, x_test = np.split(x_dataset, [-(int)(len(x_dataset) * splitting_ratio)], axis=0)
        y_train, y_test = np.split(y_dataset, [-(int)(len(y_dataset) * splitting_ratio)], axis=0)

        y_predicted = RandomForest(x_train, y_train, 100, int(len(x_train) / 3), min_in_leaf=1,
                                   boosted=boosted).predict(x_test)

        results = zip(y_predicted, y_test)
        right = 0
        for y1, y2 in results:
            # print(y1, " : ", y2, round(y1) == y2)
            if round(y1) == y2:
                right += 1

        rmse_errors.append(np.sqrt(mse(y_test, y_predicted)))
        mae_errors.append(mae(y_test, y_predicted))
        mse_errors.append(mse(y_predicted, y_test))
        accuracy.append(right / len(y_predicted) * 100)

    print('Mean Mean Absolute Error: ', sum(mae_errors) / len(mae_errors))
    print('Mean Mean Squared Error: ', sum(mse_errors) / len(mse_errors))
    print('Mean Root Mean Squared Error: ', sum(rmse_errors) / len(rmse_errors))
    print('Mean Accuracy: ', sum(accuracy) / len(accuracy))
    print('Time of calculating: ', time.time() - start_time)


# reads our datasets and run all tests
def main():
    red_wine_file_path = 'data/winequality-red.csv'
    red_dataset = pd.read_csv(red_wine_file_path, skiprows=1, sep=';', header=None)
    white_wine_file_path = 'data/winequality-white.csv'
    white_dataset = pd.read_csv(white_wine_file_path, skiprows=1, sep=';', header=None)

    print('Pramaeteres tests fo red wines')
    test_parameteres(red_dataset)

    print('Parameteres tests for white wines')
    test_parameteres(white_dataset)

    print('Default values tests for white not boosted 7:3')

    test_default_values(white_dataset, False, 0.3)

    print('Default values tests for red not boosted 7:3')

    test_default_values(red_dataset, False, 0.3)

    print('Default values tests for white  boosted 7:3')

    test_default_values(white_dataset, True, 0.3)

    print('Default values tests for red boosted 7:3')

    test_default_values(red_dataset, True, 0.3)

    print('Default values tests for white not boosted 1:1')

    test_default_values(white_dataset, False, 0.5)

    print('Default values tests for red not boosted 1:1')

    test_default_values(red_dataset, False, 0.5)

    print('Default values tests for white boosted 1:1')

    test_default_values(white_dataset, True, 0.5)

    print('Default values tests for red boosted 1:1')

    test_default_values(red_dataset, True, 0.5)


if __name__ == '__main__':
    main()
