import os
import argparse
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.aemodel import RNNModel
from torch.utils.data import TensorDataset, DataLoader


def split_booking_seq(data, input_dim: int, len_individual: int = 9):
    """
    Split csv dataset by user_id and transforms it into TorchTensor
        data: pandas dataframe,
        input_dim: dimension of input, incl. user_id,
        len_individual: max len of individual history 
    """
    unique_users = data['user_id'].unique()
    out_tensor = torch.zeros(len_individual, input_dim-1, 1)
    for user in unique_users:
        tmp_df = data[data['user_id']==user]
        tmp_df = data[data['diff_checkin'] < 9999]
        tmp_df = tmp_df[['city_id', 'diff_checkin', 'diff_inout']]
        tmp_tensor = torch.from_numpy(tmp_df.values)
        tmp_tensor = tmp_tensor.unsqueeze_(-1)

        if tmp_tensor.shape[0] > len_individual:
            tmp_tensor = tmp_tensor[:len_individual]
        elif tmp_tensor.shape[0] < len_individual:
            # pad with zeros
            target = torch.zeros(len_individual, input_dim-1,1)
            target[:tmp_tensor.shape[0],:,:] = tmp_tensor
            tmp_tensor = target

        out_tensor = torch.cat((out_tensor, tmp_tensor),2)
        print(out_tensor.shape)
    torch.save(out_tensor, "booking_tensor.pt")    
    return out_tensor


def read_file(filename, scaler, history_dim, input_dim, shuffle=True, fit=False):
    csv = pd.read_csv(filename, parse_dates=["checkin", "checkout"])
    csv = csv.sort_values(by=['user_id', 'checkin'])
    csv = csv[["user_id", "city_id", "diff_checkin", "diff_inout"]]
    assert csv.shape[1] == input_dim, "not correct input dimension"

    # if fit:
    #    csv.loc[:, "value"] = scaler.fit_transform(
    #        csv.loc[:, "value"].values.reshape(-1, 1)
    #    )
    # else:
    #    csv.loc[:, "value"] = scaler.transform(
    #        csv.loc[:, "value"].values.reshape(-1, 1)
    #    )

    seq = split_booking_seq(csv, input_dim)
    data = TensorDataset(seq, seq)
    
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
