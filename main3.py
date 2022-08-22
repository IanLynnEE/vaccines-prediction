import argparse
import time
from re import I
import torch
import pandas as pd
import numpy as np  
import sklearn                                              # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler      # noqa: F401
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

from preprocessing import read_and_encode, impute_features


def main(args):
    start = time.time()
    x_df = read_and_encode('data/training_set_features.csv')
    y_df = pd.read_csv('data/training_set_labels.csv')

    xt_df = read_and_encode('data/test_set_features.csv')
    # if args.random_missing == 'Yes':
    #     for i in  range (len(x_df.columns)):
    #         x_tmp=x_df.iloc[:, i]
    #         xt_tmp = xt_df.iloc[:, i]
    #         i1 = x_tmp.isnull().sum()
    #         i2 = xt_tmp.isnull().sum()
    #         if int(i1)>0:
    #             i3 = np.random.choice(a=x_tmp.index, size=int(i1%10))
    #             x_df.loc[i3, x_df.columns[i]] = np.nan
    print(x_df.isnull().sum())
    # print(xt_df.isnull().sum())
    if args.mode == 'evaluate':
        x_df, xt_df, y_df, yt_df = train_test_split(x_df, y_df, test_size=0.2, random_state=9)

    yyp=list()
    for i in range(1,3):
        yy_df = y_df.iloc[:, [0, i]]
        # x = x_df.drop(columns=['respondent_id', 'employment_industry', 'employment_occupation', 'health_insurance']).to_numpy()
        # xt = xt_df.drop(columns=['respondent_id', 'employment_industry', 'employment_occupation', 'health_insurance']).to_numpy()
        x = x_df.drop(columns='respondent_id').to_numpy()
        xt = xt_df.drop(columns='respondent_id').to_numpy()
        y = yy_df.drop(columns='respondent_id').to_numpy()
        y = y.ravel()
        # if i==1:
        #     x = x_df.drop(columns=['respondent_id',  'opinion_seas_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk']).to_numpy()
        #     xt = xt_df.drop(columns=['respondent_id',  'opinion_seas_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk']).to_numpy()
        # if i==2:
        #     x = x_df.drop(columns=['respondent_id', 'opinion_h1n1_sick_from_vacc', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk']).to_numpy()
        #     xt = xt_df.drop(columns=['respondent_id', 'opinion_h1n1_sick_from_vacc', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk']).to_numpy()
        features_scaler = MinMaxScaler()
        features_scaler.fit(x)
        x = features_scaler.transform(x)
        xt = features_scaler.transform(xt)
        pipline = Pipeline(steps=[('i', IterativeImputer(max_iter=20,estimator=BayesianRidge())), ('m', RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=6))])
        pipline.fit(x, y)
        yp = pipline.predict_proba(xt)
        yyp.append(yp)
    yp=yyp
    yp = pd.DataFrame(
        {
            'h1n1_vaccine': yp[0][:, 1],
            'seasonnal_vaccine': yp[1][:, 1]
        },
        index=xt_df.index
    )

    if args.mode == 'evaluate':
        yt = yt_df.drop(columns='respondent_id')
        print(roc_auc_score(yt, yp))
    else:
        yp.insert(0, 'respondent_id', xt_df['respondent_id'].astype(int))
        yp.to_csv('submission.csv', index=False)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='evaluate')
    parser.add_argument('-s', '--strategy', type=str, default='mean')
    parser.add_argument('-r', '--random_missing', type=str, default='No')
    args = parser.parse_args()

    main(args)
