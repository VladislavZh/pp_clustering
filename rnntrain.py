import os
import argparse
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.aemodel import RNNModel
from torch.utils.data import TensorDataset, DataLoader


def split_booking_seq(data, history_dim):
    len_sequence = data.shape[0]
    maxsize = len_sequence // history_dim * history_dim
    data = data[:maxsize]
    data = torch.reshape(data,(maxsize // history_dim,history_dim,data.shape[1]))
    
    return data.double()


def read_file(filename, scaler, history_dim, input_dim, shuffle=True, fit=False):
    csv = pd.read_csv(filename, parse_dates=["checkin", "checkout"])
    csv = csv.sort_values(by=['user_id', 'checkin'])
    csv = csv[["user_id", "city_id", "diff_checkin", "diff_inout"]]
    assert csv.shape[1] == input_dim

    # if fit:
    #    csv.loc[:, "value"] = scaler.fit_transform(
    #        csv.loc[:, "value"].values.reshape(-1, 1)
    #    )
    # else:
    #    csv.loc[:, "value"] = scaler.transform(
    #        csv.loc[:, "value"].values.reshape(-1, 1)
    #    )

    x = torch.from_numpy(csv.values)
    x = split_booking_seq(x, history_dim)
    data = TensorDataset(x, x)
    
    return DataLoader(data, shuffle=shuffle, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history_dim", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=30)
    parser.add_argument("--encoder_dim", type=int, default=10)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--clip", type=int, default=5)  # gradient clipping
    parser.add_argument("--learning_rate", type=float, default=0.0008)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--train_file",
        type=str,
        default="../mds20_cohortney/data/booking_challenge_tpp_labeled.csv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../mds20_cohortney/data/booking_challenge_tpp_labeled.csv",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    scaler = StandardScaler()
    train_loader = read_file(
        args.train_file, scaler, args.history_dim, args.input_dim, fit=True
    )
    test_loader = read_file(
        args.test_file,
        scaler,
        args.history_dim,
        args.input_dim,
        shuffle=False,
        fit=False,
    )

    model = RNNModel(args.input_dim, args.hidden_dim, args.encoder_dim, args.layers)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        model = model.cuda()

    for epoch in range(1, args.epochs + 1):
        print("training: epoch ", epoch)
        train_loss = 0.0
        test_loss = 0.0
        hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)

        model.train()
        for inputs, _ in train_loader:
            if train_on_gpu:
                inputs = inputs.cuda()
            if len(inputs) != batch_size:
                break

            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            optimizer.zero_grad()
            outputs, hidden1, hidden2 = model(inputs, hidden1, hidden2)
            loss = criterion(outputs, inputs)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)
        for inputs, _ in test_loader:
            if train_on_gpu:
                inputs = inputs.cuda()
            if len(inputs) != batch_size:
                break

            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            outputs, hidden1, hidden2 = model(inputs, hidden1, hidden2)
            loss = criterion(outputs, inputs)
            test_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        print(
            "Epoch: {} \tTraining Loss:{:.6f} \tValidation Loss:{:.6f}".format(
                epoch, train_loss, test_loss
            )
        )
        torch.save(
            model.state_dict(),
            os.path.join(args.checkpoint_dir, "checkpoint-%d.pth" % epoch),
        )
