import os
import argparse
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from models.aemodel import RNNModel
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/first_ae')

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


def read_file(filename: str,  history_dim: int, input_dim: int):
    """
    Takes a csv datafile, transforms to pandas dataframe,
    selects necessary features and converts to torch tensor
    """
    csv = pd.read_csv(filename, parse_dates=["checkin", "checkout"])
    csv = csv.sort_values(by=['user_id', 'checkin'])
    csv = csv[["user_id", "city_id", "diff_checkin", "diff_inout"]]
    assert csv.shape[1] == input_dim, "not correct input dimension"
    
    #csv = MinMaxScaler.fit_transform(csv)
    #seq = split_booking_seq(csv, input_dim)
    seq = torch.load('data/booking_tensor.pt')
    
    data = TensorDataset(seq, seq)
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history_dim", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=30)
    parser.add_argument("--encoder_dim", type=int, default=10)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--clip", type=int, default=5)  # gradient clipping
    parser.add_argument("--learning_rate", type=float, default=0.0008)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/booking_challenge_tpp_labeled.csv",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    scaler = MinMaxScaler()
    full_dataset = read_file(
        args.train_file, args.history_dim, args.input_dim
    )
    
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    model = RNNModel(args.input_dim, args.hidden_dim, args.encoder_dim, args.layers)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)
    if train_on_gpu:
        model = model.cuda()

    for epoch in range(1, args.epochs + 1):
        print("training: epoch ", epoch)
        train_loss = 0.0
        valid_loss = 0.0
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
        for inputs, _ in valid_loader:
            if train_on_gpu:
                inputs = inputs.cuda()
            if len(inputs) != batch_size:
                break

            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            outputs, hidden1, hidden2 = model(inputs, hidden1, hidden2)
            loss = criterion(outputs, inputs)
            valid_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        print(
            "Epoch: {} \tTraining Loss:{:.6f} \tValidation Loss:{:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )
        writer.add_scalars('Autoencoder loss', {'train': train_loss, 'validation': valid_loss}, epoch)
        torch.save(
            model.state_dict(),
            os.path.join(args.checkpoint_dir, "checkpoint-%d.pth" % epoch),
        )

    writer.close()
