import pandas as pd
import numpy as np


# random forest class holds every tree in our forest
class RandomForest:
    def __init__(self, x, y, number_of_trees, sample_size, depth=10, min_in_leaf=1, boosted=False):
        self.number_of_features = int(np.sqrt(x.shape[1]))
        self.x, self.y, self.sample_size, self.depth, self.min_in_leaf = x, y, sample_size, depth, min_in_leaf

        if boosted:
            self.trees = []
            self.trees.append(self.create_tree())
            self.y_predicted = self.predict(self.x)
            for i in range(number_of_trees - 1):
                self.trees.append(self.create_boosted_tree())
                self.y_predicted = self.predict(self.x)
        else:
            self.trees = [self.create_tree() for i in range(number_of_trees)]

    # creates new DecisionTree
    # randomly choose rows and columns to build single tree
    def create_tree(self):

        idxs = np.random.randint(low=0, high=len(self.x), size=self.sample_size)
        features_idxs = np.random.permutation(self.x.shape[1])[:self.number_of_features]

        return DecisionTree(self.x[idxs], self.y[idxs], self.number_of_features, features_idxs,
                            idxs=np.array(range(self.sample_size)), depth=self.depth, min_leaf=self.min_in_leaf)

    # creates new DecisionTree
    # randomly choose columns and with higher probability choose rows where we made bigger mistake
    def create_boosted_tree(self):

        training_results = zip(self.y_predicted, self.y)
        errors = [abs(y1 - y2) for y1, y2 in training_results]
        sum_errors = np.sum(errors)

        indexes = np.array(range(len(self.x)))
        for i in range(len(errors)):
            errors[i] = errors[i] / sum_errors

        idxs = [np.random.choice(indexes, p=errors) for i in range(self.sample_size)]
        features_idxs = np.random.permutation(self.x.shape[1])[:self.number_of_features]

        return DecisionTree(self.x[idxs], self.y[idxs], self.number_of_features, features_idxs,
                            idxs=np.array(range(self.sample_size)), depth=self.depth, min_leaf=self.min_in_leaf)

    # goes through all of trees and gets prediction
    # calculate mean value of all predictions
    def predict(self, x):
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)


# calculate standard deviation
def std_agg(count, s, sum_of_squares): return np.sqrt((sum_of_squares / count) - (s / count) ** 2)


class DecisionTree:
    def __init__(self, x, y, number_of_features, f_idxs, idxs, depth=10, min_leaf=1):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth

        self.number_of_features = number_of_features
        self.number_of_rows = len(idxs)

        # holds value for node(for root node: mean of all observations)
        self.value = np.mean(y[idxs])

        # holds how good this node splits our data
        self.score = float('inf')

        # makes our first split
        self.find_varsplit()

    # goes trough all columns and finds the best split (column's index and value's index )
    # recursively define splits for each of the further decision trees until it's reached the leaf node
    def find_varsplit(self):
        for i in self.f_idxs: self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col

        left_tree_indexes = np.nonzero(x <= self.split)[0]
        right_tree_indexes = np.nonzero(x > self.split)[0]
        left_tree_features_idxs = np.random.permutation(self.x.shape[1])[:self.number_of_features]
        right_tree_features_idxs = np.random.permutation(self.x.shape[1])[:self.number_of_features]
        self.left_tree = DecisionTree(self.x, self.y, self.number_of_features, left_tree_features_idxs,
                                      self.idxs[left_tree_indexes], depth=self.depth - 1,
                                      min_leaf=self.min_leaf)
        self.right_tree = DecisionTree(self.x, self.y, self.number_of_features, right_tree_features_idxs,
                                       self.idxs[right_tree_indexes], depth=self.depth - 1,
                                       min_leaf=self.min_leaf)

    # finds the best possible split value in certain column
    def find_better_split(self, var_idx):
        x, y = self.x[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)

        sort_y, sort_x = y[sort_idx], x[sort_idx]

        rhs_count, rhs_sum, rhs_sum2 = self.number_of_rows, sort_y.sum(), (sort_y ** 2).sum()
        lhs_count, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.number_of_rows - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_count += 1;
            rhs_count -= 1
            lhs_sum += yi;
            rhs_sum -= yi
            lhs_sum2 += yi ** 2;
            rhs_sum2 -= yi ** 2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            lhs_std = std_agg(lhs_count, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_count, rhs_sum, rhs_sum2)

            # score which is weighted average of standard deviation of two halves
            # lower score lower variance
            curr_score = lhs_std * lhs_count + rhs_std * rhs_count

            # updates score split column and split value
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_col(self):
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.value
        t = self.left_tree if xi[self.var_idx] <= self.split else self.right_tree
        return t.predict_row(xi)


# counts mean square error
def mse(h, y):
    sq_error = (h - y) ** 2
    n = len(y)
    return 1.0 / (n) * sq_error.sum()


# counts mean absolute error
def mae(h, y):
    error = abs(h - y)
    n = len(y)
    return 1.0 / n * error.sum()


# takes dataset, makes split into train and test sets
# builds our forest and then makes prediction
# measures and prints accuracy RMSE MSE and MAE
def simple_run(entry_dataset, boosted, spliting_ratio):
    dataset = entry_dataset.iloc[np.random.permutation(entry_dataset.shape[0]), :]

    x_dataset = dataset.iloc[:, 0:11].values
    y_dataset = dataset.iloc[:, 11].values

    x_train, x_test = np.split(x_dataset, [-(int)(len(x_dataset) * spliting_ratio)], axis=0)
    y_train, y_test = np.split(y_dataset, [-(int)(len(y_dataset) * spliting_ratio)], axis=0)

    # building forest and making predictions with default values
    y_predicted = RandomForest(x_train, y_train, 100, int(len(x_train) / 3), boosted=False).predict(x_test)

    results = zip(y_predicted, y_test)

    # counting right predictions
    right = 0
    for y1, y2 in results:
        # print(y1, " : ", y2, round(y1) == y2)
        if round(y1) == y2:
            right += 1

    # printing results
    print("right: ", right, ": all : ", len(y_predicted), "accuracy: ", right / len(y_predicted) * 100, '%')
    print('Mean Absolute Error:', mae(y_test, y_predicted))
    print('Mean Squared Error:', mse(y_test, y_predicted))
    print('Root Mean Squared Error:', np.sqrt(mse(y_test, y_predicted)))


def main():
    # reading datasets
    red_wine_file_path = 'data/winequality-red.csv'
    red_dataset = pd.read_csv(red_wine_file_path, skiprows=1, sep=';', header=None)
    white_wine_file_path = 'data/winequality-white.csv'
    white_dataset = pd.read_csv(white_wine_file_path, skiprows=1, sep=';', header=None)

    # run all option, boosted or not with white and red wines datasets with default values in forest
    simple_run(red_dataset, False, 0.3)
    simple_run(red_dataset, True, 0.3)
    simple_run(white_dataset, False, 0.3)
    simple_run(white_dataset, True, 0.3)


if __name__ == '__main__':
    main()
