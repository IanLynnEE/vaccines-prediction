import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler      # noqa: F401
from sklearn.metrics import roc_auc_score
import torch

from preprocessing import read_and_encode, impute_features
from models import NeuralNet


def main(args):
    x_df = read_and_encode('data/training_set_features.csv')
    y_df = pd.read_csv('data/training_set_labels.csv')

    xt_df = read_and_encode('data/test_set_features.csv')
    if args.mode == 'evaluate':
        x_df, xt_df, y_df, yt_df = train_test_split(x_df, y_df, test_size=0.2, random_state=9)
        yt = yt_df.drop(columns='respondent_id').to_numpy()

    x_df, xt_df = impute_features(x_df, xt_df, strategy='mean', known_test=False)

    x = x_df.drop(columns='respondent_id').to_numpy()
    xt = xt_df.drop(columns='respondent_id').to_numpy()
    y = y_df.drop(columns='respondent_id').to_numpy()

    features_scaler = MinMaxScaler()
    features_scaler.fit(x)
    x = features_scaler.transform(x)
    xt = features_scaler.transform(xt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    model_h1n1 = NeuralNet(35, 70, 70, 2).to(device)
    model_seasonal = NeuralNet(35, 70, 70, 2).to(device)

    optimizer = torch.optim.Adam(model_h1n1.parameters(), lr=args.learning_rate)
    train(x, y[:, 0], xt, yt[:, 0],  model_h1n1, criterion, optimizer, device, args.epochs)
    yp0 = predict_proba(xt, model_h1n1, device)

    optimizer = torch.optim.Adam(model_seasonal.parameters(), lr=args.learning_rate)
    train(x, y[:, 1], xt, yt[:, 1], model_seasonal, criterion, optimizer, device, args.epochs)
    yp1 = predict_proba(xt, model_seasonal, device)

    yp = pd.DataFrame(
        {
            'h1n1_vaccine': yp0[:, 1],
            'seasonal_vaccine': yp1[:, 1]
        },
        index=xt_df.index
    )

    if args.mode == 'evaluate':
        yt = yt_df.drop(columns='respondent_id')
        print(roc_auc_score(yt, yp))
    else:
        yp.insert(0, 'respondent_id', xt_df['respondent_id'].astype(int))
        yp.to_csv('submission.csv', index=False)


def train(x, y, xv, yv, model, criterion, optimizer, device, epochs) -> tuple[list, list]:
    t_features = torch.from_numpy(x).to(device).float()
    t_labels = torch.from_numpy(y).to(device)
    t_loss = [0 for _ in range(epochs)]
    v_features = torch.from_numpy(xv).to(device).float()
    v_labels = torch.from_numpy(yv).to(device)
    v_loss = [0 for _ in range(epochs)]

    for epoch in range(epochs):
        model.train()
        output = model(t_features)
        loss = criterion(output, t_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        output = model(v_features)
        v_loss[epoch] = criterion(output, v_labels)
        t_loss[epoch] = loss
    return t_loss, v_loss
        

def predict_proba(xt, model, device) -> np.ndarray:
    with torch.no_grad():
        features = torch.from_numpy(xt).to(device)
        prob = model(features.float())
    return prob.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='evaluate')
    parser.add_argument('-s', '--strategy', type=str, default='mean')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    args = parser.parse_args()

    main(args)
