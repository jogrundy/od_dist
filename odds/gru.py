#GRU coded up to look for outliers.

import os
from time import time
import platform

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# import gc

#started with code from https://blog.floydhub.com/gru-with-pytorch/
#then massively refactored.



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.criterion=nn.MSELoss()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.criterion=nn.MSELoss()

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def train(train_loader, learn_rate, device, batch_size, in_dim, out_dim, hidden_dim=256, EPOCHS=5, model_type="GRU"):


    # Setting common hyperparameters
    input_dim = in_dim
    output_dim = out_dim
    n_layers = 2

    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers, device)
    model.to(device)

    # Defining loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


    model.train()
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = model.criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        t1 = time()-start_time

        epoch_times.append(t1)
    if device =='cuda':
        # gc.collect()
        torch.cuda.empty_cache()
    tt = sum(epoch_times)

    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    t1 = time() - start_time
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)

    return outputs, targets, sMAPE

def model_eval(model, data_loader, targets, scaler, device):
    """
    uses MSE to evaluate model
    """
    model.eval()
    model.cpu()

    batch_size, tw, p =next(iter(data_loader))[0].shape
    t0 = time()
    preds = []
    scores = []
    targets = []
    with torch.no_grad():
        for input, label in data_loader:

            # print(inputs.shape, targets.shape)
            h = model.init_hidden(batch_size)

            out, _ = model(input.float(), h)
            loss = model.criterion(out, label)

            pred = out.detach().numpy()[0]
            label = label.detach().numpy()[0]
            score = loss.detach().numpy().item(0)

            scores.append(score)
            preds.append(pred)
            targets.append(label)


    return preds, targets, scores



def compile_data(data, tw, pad=False, test_split = 0.2):
    """
    assumes data is n,p numpy matrix.
    produces set of serieses with or without padding
    """
    n,p = data.shape
    inputs = np.zeros((n,tw,p))
    targets = np.zeros((n, p))
    if pad:
        data = np.concatenate([np.zeros((tw,p)), data])
    for i in range(tw, n+tw):
        inputs[i-tw] = data[i-tw:i, :]
        targets[i-tw] = data[i,:]
    inputs = inputs.reshape(-1,tw,p)
    targets = targets.reshape(-1,p)

    # Split data into train/test portions and combining all data from different files into a single array
    test_ind = int(test_split*len(inputs))
    train_x = inputs[:-test_ind, :, :]
    train_y = targets[:-test_ind, :]
    test_x = inputs[-test_ind:]
    test_y = targets[-test_ind:]

    return train_x, train_y, inputs, targets

def ese(pred, target):
    """
    takes in predicted values and actual values, returns elementwise squared error
    via (x-y)^2
    """
    errs = (np.subtract(pred, target))**2
    errs = np.sum(errs, axis=1)
    return np.sqrt(errs)

def get_GRU_os(X):
    n,p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(n//4) > 16:
        tw = 16
    else:
        tw = int(n//8)
    test_p = .2

    if n<8:
        batch_size=1
    elif n<16:
        batch_size=4
    elif n<64:
        batch_size=16
    elif n<128:
        batch_size=32
    else:
        batch_size = 128

    scaler = None # works better without scaler. data may need to be scaled before use.

    train_x, train_y, inputs, targets = compile_data(X, tw, pad=True, test_split = test_p)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(data, shuffle=False, batch_size=1, drop_last=True)
    lr = 0.001
    gru_model = train(train_loader, lr, device, batch_size, p, p, hidden_dim=64, EPOCHS=20, model_type="GRU")
    gru_preds, targets_scaled, score = model_eval(gru_model, data_loader, targets, None, device)

    return score

def get_LSTM_os(X):
    n,p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(n//4) > 16:
        tw = 16
    else:
        tw = int(n//8)
    test_p = 0.2

    if n<8:
        batch_size=1
    elif n<16:
        batch_size=4
    elif n<64:
        batch_size=16
    elif n<128:
        batch_size=32
    else:
        batch_size = 128

    scaler = None
    in_dim = p
    out_dim = p


    train_x, train_y, inputs, targets = compile_data(X, tw, pad=True, test_split = test_p)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(data, shuffle=False, batch_size=1, drop_last=True)
    lr = 0.001
    lstm_model = train(train_loader, lr, device, batch_size, in_dim, out_dim,
                        hidden_dim=64, EPOCHS=20, model_type="LSTM")
    lstm_preds, targets_scaled, score = model_eval(lstm_model, data_loader, targets, scaler, device)

    return score
