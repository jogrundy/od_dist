# VAE on MNIST
import torch
import torchvision
from torch import nn
from torch import optim
# import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib.pyplot as plt
from ae import SynDataset
import argparse


# def to_img(x):
#     x = x.clamp(0, 1)
#     x = x.view(x.size(0), 1, 28, 28)
#     return x


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


        losses.append(loss)
    # print(np.array(losses, dtype='float')[:10])
    losses = np.array(losses, dtype='float')
        # np.savetxt(loss_fname, losses)
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

if __name__ == '__main__':
    from torchvision.datasets import MNIST
    os.makedirs("vae_img", exist_ok=True)

    num_epochs = 2
    batch_size = 128
    learning_rate = 1e-3

    img_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss(size_average=False)
    layers = [784,400,20]
    model = VAE(layers, criterion)
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    # reconstruction_function = nn.MSELoss(size_average=False)





    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            optimizer.zero_grad()
            output, mu, logvar = model(img)
            loss = model.loss_fn(img, output, mu, logvar)
            loss.backward()
            train_loss += loss.data
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    loss.data / len(img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        if epoch % 10 == 0:
            save = to_img(output.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))

    dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False)
    losses = []

    for i , data in enumerate(dataloader_test):

        img, o = data
        # print(i, o)
        img = img.view(img.size(0), -1)
        img = Variable(img)
        # ===================forward=====================
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = model.loss_fn(recon_batch, img, mu, logvar)

        losses.append(loss)
    losses = np.array(losses)
    k = 5
    worst = losses.argsort()[-k:][::-1]
    plt.figure(figsize=(8,3))
    for i in range(k):
        idx = worst[i]
        plt.subplot(1,k,i+1)
        img = dataset[idx][0].reshape(28,28)
        cls = dataset[idx][1]
        plt.axis('off')
        plt.imshow(img, cmap='Greys')

        plt.title('loss={:.1f}, {}'.format(losses[idx], cls))
    plt.savefig('./images/highest_losses_vae.eps',bbox_inches='tight')
    plt.show()
