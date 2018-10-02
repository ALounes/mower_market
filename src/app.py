# TODO : adding modules

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import re
import math


def rmsle(predictions, real_values):
    """
    RMSLE --> Root mean square logarithmic error
    :param predictions: array of prediction values
    :param real_values: array of real values
    :return: rmsle value
    """
    sum = 0
    for i in range(0, len(predictions)):
        sum += math.pow(math.log(float(predictions[i]) + 1) - math.log(float(real_values[i]) + 1), 2)

    return float(math.sqrt(sum / len(predictions)))


# TODO : deprecated, Use Cross validation
def get_the_best_argument(x_train, x_test, y_train, y_test):
    """

    :param x_train: input training set
    :param x_test: input testing set
    :param y_train: output training set
    :param y_test: output testing set
    :return: the minimum error and the best configuration fo the n_components parameter
    """

    min = 1
    indice = 0

    for i in range(1, 14):
        # Configuration de la PLS
        reg2 = cross_decomposition.PLSRegression(n_components=i, scale=True)
        # regression training
        reg2.fit_transform(x_train, y_train)
        # prediction
        y_pred = reg2.predict(x_test)

        if min > rmsle(y_pred, y_test):
            min = rmsle(y_pred, y_test)
            indice = i

        # print(rmsle(y_pred, y_test))

    return min, indice


# TODO : should be in a Sklearn pipeline, adding normalization
def cleaning_dataframe(df):
    """
    Cleaning the df dataframe
    :param df:
    :return:
    """
    # mettre les missing value en NAN et le reste en float
    df.prod_cost = pd.to_numeric(df.prod_cost, errors='coerce')

    # Virer les doublon et la colonne avec un NAN
    df = df.drop_duplicates(subset=['id']).dropna()

    # Supression des outliers <= 0
    df = df.loc[df['prod_cost'] > 0]

    # Cleaning warrenty rows
    df.warranty = df.warranty.apply(lambda x: re.findall('\d', x)[0] + '_ans')

    # creation de variables
    df = pd.get_dummies(df, columns=['product_type', 'quality', 'warranty'])

    return df


# TODO : refactor, adding parameters to have a generic function
def linear_training(df, var):
    """

    :param df:
    :param var:
    :return:
    """

    condition = df[var] == 1
    filtred_df = df[condition]

    y = filtred_df.attractiveness.values.reshape(-1, 1)
    X = filtred_df.drop(['id', 'market_share', 'attractiveness'], axis=1).values

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    return reg


# TODO : need to be refactored
def linear_prediction(df, reg, var):
    """

    :param df:
    :param var:
    :return:
    """
    condition = df[var] == 1
    filtred_df = df[condition]

    ids = filtred_df.id.values.reshape(-1, 1)
    X = filtred_df.drop(['id'], axis=1).values

    prediction = reg.predict(X)

    res_df = pd.DataFrame(ids, columns=['id'])
    res_df['attractiveness'] = prediction

    return res_df


# TODO: refactor to loop from a dictionary and return an array or dict
def get_regression_model(df):
    """

    :param df:
    :return:
    """
    reg1 = linear_training(df, 'product_type_auto-portee')
    reg2 = linear_training(df, 'product_type_electrique')
    reg3 = linear_training(df, 'product_type_essence')

    return reg1, reg2, reg3


# TODO: refactor to loop from a dictionary
def get_prediction(df, reg_auto_portee, reg_electrique, reg_essence):
    """

    :param df:
    :param reg_auto_portee:
    :param reg_electrique:
    :param reg_essence:
    :return:
    """

    auto_porte_prediction = linear_prediction(df, reg_auto_portee, 'product_type_auto-portee')
    electrique_prediction = linear_prediction(df, reg_electrique, 'product_type_electrique')
    essence_prediction = linear_prediction(df, reg_essence, 'product_type_essence')

    return pd.concat([auto_porte_prediction, electrique_prediction, essence_prediction])


# TODO : Should implement a pipeline
def main():
    """
    The main function
    """
    # TODO : the file name should be passed into a parameter
    training_df = pd.read_csv('../data/mower_market_snapshot.csv', sep=';')
    submission_df = pd.read_csv('../data/submission_set.csv', sep=';')

    # Cleaning dataframe
    training_df = cleaning_dataframe(training_df)
    submission_df = cleaning_dataframe(submission_df)

    # TODO : should implement a classification logic then the regression ...
    reg_auto_portee, reg_electrique ,reg_essence = get_regression_model(training_df)
    prediction_df = get_prediction(submission_df, reg_auto_portee, reg_electrique, reg_essence)

    # Write result into csv file
    # TODO : the file name should be a parameter
    prediction_df.to_csv('../data/achab_lounes_attractiveness.csv', index=False)


def evaluate_prediction():
    """

    :return:
    """
    df = pd.read_csv('../data/mower_market_snapshot.csv', sep=';')

    training_df, submission = train_test_split(df, test_size=0.2)

    test_df = submission.loc[:, ['id', 'attractiveness']]
    submission_df = submission.drop(['market_share', 'attractiveness'], axis=1)

    # Cleaning dataframe
    training_df = cleaning_dataframe(training_df)
    submission_df = cleaning_dataframe(submission_df)

    reg_auto_portee, reg_electrique, reg_essence = get_regression_model(training_df)

    prediction_df = get_prediction(submission_df, reg_auto_portee, reg_electrique, reg_essence)

    res = pd.DataFrame(pd.merge(test_df, prediction_df, on='id'))

    print(rmsle(res.attractiveness_x.values.reshape(-1, 1), res.attractiveness_y.values.reshape(-1, 1)))


if __name__ == '__main__':
    main()
