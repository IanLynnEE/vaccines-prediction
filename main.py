import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    h1n1 = train_df['h1n1_vaccine']
    seasonal = train_df['seasonal_vaccine']
    print(np.count_nonzero(h1n1 != seasonal))
    print(np.count_nonzero(h1n1 > seasonal))
    print(np.count_nonzero(h1n1 < seasonal))
    print(train_df.shape)

    # On test set, we may need the result of h1n1/seasonal before predicting seasonal/h1n1, depending on the setting.
    # The following function will try to insert the prediction result of h1n1/seasonal into test_df if needed.
    if insert_predict_result(test_df, args) == False:
        print('Fail to insert prediction result. Program aborted.')
        return

    x, y, feature_names = select_prediction_target(train_df, vaccine=args.vaccine)
    xt, yt, _ = select_prediction_target(test_df, vaccine=args.vaccine)

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_features='sqrt',
        n_jobs=6
    )
    clf.fit(x, y)
    yp = clf.predict_proba(xt)[:,1]

    # fig = plt.figure()
    # importances = pd.Series(clf.feature_importances_, index=feature_names)
    # importances.plot.bar()
    # plt.title(f'{args.vaccine}\nwith h1n1 as ground truth')
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel('Features')
    # plt.ylabel('Importances')
    # fig.subplots_adjust(bottom=0.3)
    # plt.show()

    if args.mode == 'evaluate':
        print(f'AUROC for {args.vaccine} : {roc_auc_score(yt, yp)}')
    else:
        write_result(yp, args.vaccine)

    # Training cascade without ground truth
    if args.vaccine == 'seasonal':
        new_feat = clf.predict_proba(x)[:, 1]
        x, y, _ = select_prediction_target(train_df, vaccine='h1n1')
        xt, yt, _ = select_prediction_target(test_df, vaccine='h1n1')
        x = np.column_stack((x, new_feat))
        xt = np.column_stack((xt, yp))
    elif args.vaccine == 'h1n1':
        new_feat = clf.predict_proba(x)[:, 1]
        x, y, _ = select_prediction_target(train_df, vaccine='seasonal')
        xt, yt, _ = select_prediction_target(test_df, vaccine='seasonal')
        x = np.column_stack((x, new_feat))
        xt = np.column_stack((xt, yp))
    else:
        return

    clf.fit(x, y)
    yp = clf.predict_proba(xt)[:, 1]

    # plt.clf()
    # fig = plt.figure()
    # feature_names = feature_names.append(pd.Index([args.vaccine]))
    # importances = pd.Series(clf.feature_importances_, index=feature_names)
    # importances.plot.bar()
    # plt.title(f'seasonal_from_h1n1\nwithout using ground truth of h1n1')
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel('Features')
    # plt.ylabel('Importances')
    # fig.subplots_adjust(bottom=0.3)
    # plt.show()

    if args.mode == 'evaluate':
        print(f'AUROC for using {args.vaccine} as a feature: {roc_auc_score(yt, yp)}')
    else:
        write_result(yp, 'seasonal')
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
        x = data.drop(columns=['h1n1_vaccine', 'seasonal_vaccine'])
    elif vaccine == 'seasonal':
        # Train with standard features
        y = data['seasonal_vaccine'].to_numpy()
        x = data.drop(columns=['h1n1_vaccine', 'seasonal_vaccine'])
    elif vaccine == 'h1n1_from_seasonal':
        # Train with the result of seasonal
        y = data['h1n1_vaccine'].to_numpy()
        x = data.drop(columns=['h1n1_vaccine'])
    elif vaccine == 'seasonal_from_h1n1':
        # Train with the result of h1n1
        y = data['seasonal_vaccine'].to_numpy()
        x = data.drop(columns=['seasonal_vaccine'])
    return x.to_numpy(), y, x.columns


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


def write_result(yp, vaccine):
    old_data = pd.DataFrame()
    if os.path.exists('submission.csv'):
        old_data = pd.read_csv('submission.csv')
    if vaccine == 'h1n1' or vaccine == 'h1n1_from_seasonal':
        if 'h1n1_vaccine' in old_data.columns:
            if input('Overwrite the result of the h1n1 prediction? [y/n]: ') != 'y':
                return
        old_data['h1n1_vaccine'] = yp
    if vaccine == 'seasonal' or vaccine == 'seasonal_from_h1n1':
        if 'seasonal_vaccine' in old_data.columns:
            if input('Overwrite the result of the seasonal prediction? [y/n]: ') != 'y':
                return
        old_data['seasonal_vaccine'] = yp
    old_data.to_csv('submission.csv', index=False)
    return


if __name__ == '__main__':
    main()
