import argparse

import pandas as pd
import numpy as np                                                  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler      # noqa: F401
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from preprocessing import read_and_encode, impute_features


def main(args):
    x_df = read_and_encode('data/training_set_features.csv')
    y_df = pd.read_csv('data/training_set_labels.csv')

    xt_df = read_and_encode('data/test_set_features.csv')
    if args.mode == 'evaluate':
        x_df, xt_df, y_df, yt_df = train_test_split(x_df, y_df, test_size=0.2, random_state=9)

    x_df, xt_df = impute_features(x_df, xt_df, strategy='mean', known_test=False)

    x = x_df.drop(columns='respondent_id').to_numpy()
    xt = xt_df.drop(columns='respondent_id').to_numpy()
    y = y_df.drop(columns='respondent_id').to_numpy()

    features_scaler = MinMaxScaler()
    features_scaler.fit(x)
    x = features_scaler.transform(x)
    xt = features_scaler.transform(xt)

    # RandomForestClassifier supports multi-output classification.
    clf = RandomForestClassifier(
        n_estimators=1000,
        max_features='sqrt',
        n_jobs=6
    )
    clf.fit(x, y)
    yp = clf.predict_proba(xt)

    yp = pd.DataFrame(
        {
            'h1n1_vaccine': yp[0][:, 1],
            'seasonal_vaccine': yp[1][:, 1]
        },
        index=xt_df.index
    )

    if args.mode == 'evaluate':
        yt = yt_df.drop(columns='respondent_id')
        print(roc_auc_score(yt, yp))
    else:
        yp.insert(0, 'respondent_id', xt_df['respondent_id'].astype(int))
        yp.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='evaluate')
    parser.add_argument('-s', '--strategy', type=str, default='mean')
    args = parser.parse_args()

    main(args)
