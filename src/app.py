import pandas as pd
import re


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

    sum = sum * (1/len(predictions))

    return float(math.sqrt(sum))


def main():
    """

    :return:
    """

    training_df = pd.read_csv('../data/mower_market_snapshot.csv', sep=';')
    submission_df = pd.read_csv('../data/submission_set.csv', sep=';')

    # mettre les missing value en NAN et le reste en float
    training_df.prod_cost = pd.to_numeric(training_df.prod_cost, errors='coerce')

    # Virer les doublon et la colonne avec un NAN
    training_df = training_df.drop_duplicates(subset=['id']).dropna()

    # Supression des outliers <= 0
    training_df = training_df.loc[training_df['prod_cost'] > 0]

    # Cleaning warrenty rows
    training_df.warranty = training_df.warranty.apply(lambda x: re.findall('\d', x)[0] + '_ans')

    # creation de variables
    training_df = pd.get_dummies(training_df, columns=['product_type', 'quality', 'warranty'])


if __name__ == '__main__':
    main()
