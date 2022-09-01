import argparse
import os

import pandas as pd
import numpy as np                                                  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler      # noqa: F401
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from preprocessing import read_and_encode, impute_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='evaluate')
    parser.add_argument('-s', '--strategy', type=str, default='mean')
    parser.add_argument('--vaccine', type=str, default='h1n1')
    args = parser.parse_args()

    train_df, test_df = get_data(args)

    # On test set, we may need the result of h1n1/seasonal before predicting seasonal/h1n1, depending on the setting.
    # The following function will try to insert the prediction result of h1n1/seasonal into test_df if needed.
    if insert_predict_result(test_df, args) == False:
        return

    x, y = select_prediction_target(train_df, vaccine=args.vaccine)
    xt, yt = select_prediction_target(test_df, vaccine=args.vaccine)

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_features='sqrt',
        n_jobs=6
    )
    clf.fit(x, y)
    yp = clf.predict_proba(xt)
    if args.mode == 'evaluate':
        print(roc_auc_score(yt, yp[:, 1]))
    else:
        write_result(yp[:, 1], args)
    return


def get_data(args) -> tuple:
    x = read_and_encode('data/training_set_features.csv')
    xt = read_and_encode('data/test_set_features.csv')
    y = pd.read_csv('data/training_set_labels.csv')
    y = y.drop(columns='respondent_id')

    cols = x.columns.union(y.columns, sort=False).difference(['respondent_id'], sort=False)

    if args.mode == 'evaluate':
        x, xt, y, yt = train_test_split(x, y, test_size=0.2)#, random_state=9)

    x, xt = impute_features(x, xt, strategy='mean', known_test=False)

    x.drop(columns='respondent_id', inplace=True)
    xt.drop(columns='respondent_id', inplace=True)

    features_scaler = MinMaxScaler()
    features_scaler.fit(x)
    x = features_scaler.transform(x)
    xt = features_scaler.transform(xt)

    train = pd.DataFrame(np.hstack((x, y.to_numpy())), columns=cols, )
    if args.mode == 'evaluate':
        test = pd.DataFrame(np.hstack((xt, yt.to_numpy())), columns=cols)
    else:
        test = pd.DataFrame(xt, columns=cols.difference(['h1n1_vaccine', 'seasonal_vaccine'], sort=False))
    return train, test


def select_prediction_target(data, vaccine):
    if vaccine == 'h1n1':
        # Train with standard features
        y = data['h1n1_vaccine'].to_numpy()
        x = data.drop(columns=['h1n1_vaccine', 'seasonal_vaccine']).to_numpy()
    elif vaccine == 'seasonal':
        # Train with standard features
        y = data['seasonal_vaccine'].to_numpy()
        x = data.drop(columns=['h1n1_vaccine', 'seasonal_vaccine']).to_numpy()
    elif vaccine == 'h1n1_from_seasonal':
        # Train with the result of seasonal
        y = data['h1n1_vaccine'].to_numpy()
        x = data.drop(columns=['h1n1_vaccine']).to_numpy()
    elif vaccine == 'seasonal_from_h1n1':
        # Train with the result of h1n1
        y = data['seasonal_vaccine'].to_numpy()
        x = data.drop(columns=['seasonal_vaccine']).to_numpy()
    return x, y


def insert_predict_result(test_df, args):
    if args.mode == 'evaluate':
        return True
    test_df['h1n1_vaccine'] = 0
    test_df['seasonal_vaccine'] = 0
    if args.vaccine == 'h1n1' or args.vaccine == 'seasonal':
        return True
    if not os.path.exists('submission.csv'):
        print('Cannot read submission.csv file. Please check your configuration.')
        return False
    yp = pd.read_csv('submission.csv')
    if args.vaccine == 'h1n1_from_seasonal':
        test_df['seasonal_vaccine'] = yp['seasonal_vaccine']
    if args.vaccine == 'seasonal_from_h1n1':
        test_df['h1n1_vaccine'] = yp['h1n1_vaccine']
    return True


def write_result(yp, args):
    old_data = pd.DataFrame()
    if os.path.exists('submission.csv'):
        old_data = pd.read_csv('submission.csv')
    if args.vaccine == 'h1n1' or args.vaccine == 'h1n1_from_seasonal':
        if 'h1n1_vaccine' in old_data.columns:
            if input('Overwrite the result of the h1n1 prediction? [y/n]: ') != 'y':
                return
        old_data['h1n1_vaccine'] = yp
    if args.vaccine == 'seasonal' or args.vaccine == 'seasonal_from_h1n1':
        if 'seasonal_vaccine' in old_data.columns:
            if input('Overwrite the result of the seasonal prediction? [y/n]: ') != 'y':
                return
        old_data['seasonal_vaccine'] = yp
    old_data.to_csv('submission.csv', index=False)
    return


if __name__ == '__main__':
    main()
