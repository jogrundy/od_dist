"""
builds VAE and provides outlier score with get_VAE_os
"""
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
from .ae import SynDataset


class VAE(nn.Module):
    def __init__(self, layers, criterion):
        super(VAE, self).__init__()
        ls = []
        for i in range(len(layers)-2):
            ls.append(nn.Linear(layers[i], layers[i+1]))
            ls.append(nn.ReLU(True))

        self.pre_encoder = nn.Sequential(
        *ls
        )
        self.encode_mu = nn.Linear(layers[-2], layers[-1])
        self.encode_sig = nn.Linear(layers[-2], layers[-1])

        ls = []
        for i in range(len(layers)-1,1, -1):
            # print(layers[i])
            ls.append(nn.Linear(layers[i], layers[i-1]))
            ls.append(nn.ReLU(True))
        ls.append(nn.Linear(layers[1], layers[0]))
        ls.append(nn.Softmax(dim=0))

        self.decoder = nn.Sequential(
        *ls
        )
        self.criterion = criterion

    def encode(self, x):
        h = self.pre_encoder(x)
        return self.encode_mu(h), self.encode_sig(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps) #.to(self.device)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


    def loss_fn(self, x,output, mu, logvar):
        """
        recon_x: generated images
        x: original images
        mu: latent mean
        logvar: latent log variance
        """
        BCE = self.criterion(output, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        # print(KLD)
        # print(BCE)
        # raise
        return BCE + KLD

def train_dataset(model, loader, optimizer, params, device):
    n,p,dummy, dummy, dummy, dummy, dummy, num_epochs = params
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = []

        for batch_idx, data in enumerate(loader):
            data = Variable(data[0]).to(device)
            output, mu, logvar = model(data)
            loss = model.loss_fn(data, output, mu, logvar)
            loss.backward()
            train_loss.append( loss.data)

            optimizer.zero_grad()
            optimizer.step()


    return model, train_loss

def get_losses(model, dataset, params, device):
    """
    calculates reconstruction loss for each datapoint
    """

    n,p,r, p_frac, p_quant,gamma, ta, num_epochs = params
    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    losses = []
    for i,data in enumerate(loader):
        data = Variable(data).to(device)
        # ===================forward=====================
        output, mu , logvar = model(data)
        loss = model.loss_fn(data, output, mu, logvar)
        #======== Get outlier score =================#
        losses.append(loss.detach().cpu().numpy())

    losses = np.array(losses, dtype='float')

    return losses

def get_vae_losses(X):
    """
    trains vae on np array X, returns reconstruction loss for each data sample
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs=50
    batch_size = 8
    learning_rate = 0.001
    dataset=SynDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n = X.shape[0]
    p = X.shape[1]
    ta = -1
    dummy = -1

    params = (n,p,dummy, dummy, dummy, dummy, dummy, num_epochs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    i_layer_size = p
    h1_layer_size = 64
    e_layer_size = 8
    layers = [i_layer_size, h1_layer_size, e_layer_size]
    # label = 'VAE'

    model = VAE(layers, criterion)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, train_loss = train_dataset(model, loader, optimizer, params, device)

    losses = get_losses(model, dataset, params, device)
    return losses

def get_VAE_os(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    gives reconstruciton error from AE, should be largest for outliers
    """
    losses = get_vae_losses(X)
    return losses
