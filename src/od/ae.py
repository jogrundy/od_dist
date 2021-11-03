#put IMS data through autoencoders

import numpy as np
import matplotlib.pyplot as plt
import os
import platform
# from test_FP import generate_test, ta_1, ta_2, ta_3



import torch
from torch.utils.data import DataLoader, Dataset
# import torchvision
from torch import nn
from torch.autograd import Variable
import numpy.linalg as la
import matplotlib.cm as cm
from time import time, localtime
from torchvision import transforms

data_path = os.path.expanduser('~') +'/Data/synthetic/'

class AE(nn.Module):
    def __init__(self, layers):
        super(AE, self).__init__()

        #puts all the layers as specified by the list of layer sizes 'layers'
        #in to a list of pytorch layers with ReLU activation
        ls = []
        for i in range(len(layers)-2):
            ls.append(nn.Linear(layers[i], layers[i+1]))
            ls.append(nn.ReLU(True))
        ls.append(nn.Linear(layers[len(layers)-2], layers[len(layers)-1]))
        #splats the layers in to the encoder.
        self.encoder = nn.Sequential(*ls)
        #puts all the layers as specified by the list of layer sizes 'layers'
        #in to an upside down list of pytorch layers with ReLU activation
        ls = []
        for i in range(len(layers)-1,1, -1):
            # print(layers[i])
            ls.append(nn.Linear(layers[i], layers[i-1]))
            ls.append(nn.ReLU(True))
        ls.append(nn.Linear(layers[1], layers[0]))
        #splats the layers in to the decoder.
        self.decoder = nn.Sequential(
            *ls,
            nn.Tanh()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)




class SynDataset(Dataset):
    """
    Uses synthetic data to feed dataloader for pytorch
    """
    def __init__(self, X, overlap=None):
        """
        Initialization, X is data set n data points by p dimensions
        """
        super(SynDataset).__init__()
        self.data = X

    def __len__(self):
        """
        Denotes the total number of points
        """
        return self.data.shape[0]

    def generate_data(self): #do i need to do this separately?
        """
        takes dataset and returns generator
        """
        for i in range(self.data.shape[0]):
            yield torch.tensor(self.data[i,:], dtype=torch.float)


    def __iter__(self):
        """
        uses a generator for the IMS data set
        """
        return self.generate_data()

    def __getitem__(self, index):
        """
        required for a shuffle, first work out which second to get,
        then how far along we need to be
        """
        sample = self.data[index,:]
        return torch.tensor(sample.T, dtype=torch.float)

def train_dataset(model, loader, criterion, optimizer, num_epochs, PATH, device):


    losses = []
    training_losses = []

    t0 = time()
    for epoch in range(num_epochs):
        t1 = time()

        for i,data in enumerate(loader):

            data = Variable(data).to(device)
            # ===================forward=====================

            output = model(data)
            loss = criterion(output, data)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        training_losses.append(loss.data)
        t2 = time() - t1
        tr = (num_epochs - (epoch +1))*t2

    training_losses = np.array(training_losses)

    return model, training_losses





def get_codes(model, params, dataset):
    """
    """
    n,p,r, p_frac, p_quant,gamma, ta = params

    fname=data_path+'X_code_{}_{}_{}_{}_{}_{}_{}.txt'.format(
            n,p,r, p_frac, p_quant,gamma, ta)
    try:
        X_code = np.loadtxt(fname)
        if X_code.shape[0] == 0:
            print('trying again..')
            raise

    except:
        print('calculating codes.. ')
        codes = []
        for i, data in enumerate(dataset):
            if i %100==0:
                print('i = {}'.format(i))
            data = Variable(torch.tensor(data, dtype=torch.float))
            # ===================forward=====================
            code = model.encode(data)
            code = code.detach().numpy()
            codes.append(code)

        X_code = np.array(codes)
        print(X_code.shape)
        np.savetxt(fname, X_code)
    return X_code



def pca_proj(M, params, show=True):
    """
    takes in n,p matrix, returns projection matrix to n,k matrix
    using principle component analysis
    """
    n,p,r, p_frac, p_quant,gamma, ta = params
    # M = mean_centering(M) # n,p matrix
    k=2
    n,p = M.shape
    U, S, VT = la.svd(M, full_matrices=False) # gives U (n,p) ,S(p,), V(p, p)
    pa =U[:k,:]  #p,p matrix to p,k , take first k principal eig vecs cols of V
    Xp = np.dot(M, pa.T)
    c = np.arange(Xp.shape[0])

    fig = plt.figure()
    for i in range(n):
        if i%1000 == 0:
            print(i)
        plt.plot([Xp[i,0]],[Xp[i,1]], '.', color = cm.rainbow(c[i]/n))
    fname = './images/Synth_ae_expt_ta_{}_plot.eps'.format(ta)
    plt.ylabel('PC 2')
    plt.xlabel('PC 1')

    plt.title('Projection on to Principal Components')
    plt.savefig(fname)
    if show:
        plt.show()

def get_losses(model, dataset, criterion, device):
    """
    calculates loss for each item, here a sample from a seconds worth of data
    """

    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    losses = []
    for i,data in enumerate(loader):
        data = Variable(data).to(device)
        # ===================forward=====================
        output = model(data)
        loss = criterion(data, output)
        losses.append(loss)
    losses = np.array(losses, dtype='float')
        # np.savetxt(loss_fname, losses)
    return losses

def get_grid_losses(train_data, grid_data):

    model, dataset, criterion, device = get_model(train_data)
    losses = get_losses(model, grid_data, criterion, device)
    return losses

def get_model(X):

    n = X.shape[0]
    p = X.shape[1]

    sys = platform.system()

    # print("sys = {}".format(sys))

    # Look for available gpu with cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print('Device = {}'.format(device))

    num_epochs = 50
    batch_size = 8
    learning_rate = 0.001


    i_layer_size = p
    h1_layer_size = 64
    e_layer_size = 8
    layers = [i_layer_size, h1_layer_size, e_layer_size]
    label = 'AE'

    dataset = SynDataset(X)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    model = AE(layers).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    PATH = './synth_{}_{}_{}_{}_autoencoder.pth'.format(
            num_epochs, i_layer_size, h1_layer_size,  e_layer_size)
    model, training_losses = train_dataset(model, loader, criterion, optimizer, num_epochs, PATH, device)
    return model, dataset, criterion, device

def get_ae_losses(X):
    model, dataset, criterion, device = get_model(X)
    losses = get_losses(model, dataset, criterion, device)
    return losses

def get_AE_os(X):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    higher score = more outlier
    """
    losses = get_ae_losses(X)
    #gives reconstruciton error from AE, should be largest for outliers
    return losses

if __name__ == '__main__':

     data_path = os.path.expanduser('~') +'/Data/synthetic/'
#
# x = np.arange(len(losses))
# plt.figure()
# plt.plot(x, losses)
# plt.xlabel('Running time')
# plt.ylabel('reconstruction loss')
# title = 'Reconstruction loss for ta {}'.format(ta)
# plt.title(title)
# fname = './images/synth_ta_{}_loss_plot.eps'.format(ta)
# # plt.show()
#
# x = np.arange(len(training_losses))
# plt.figure()
# plt.plot(x, training_losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# title = 'Training Loss on Synth data ta_{}'.format(ta)
# plt.title(title)
# fname = './images/Synth_ta{}_training_loss_plot.eps'.format(ta)
# plt.savefig(fname)
# # plt.show()
#
# n_samp = 1
# batchsize = 1
# params = (n,p,r, p_frac, p_quant,gamma, ta)
# X_code = get_codes(model, params, dataset)
# print('X_code shape = {}, {}'.format(X_code.shape[0], X_code.shape[1]))
# pca_proj(X_code, params)

# X_gen = generator(*params)
# X = next(X_gen)
# for i,data in enumerate(X_gen):
#     continue
# print(i)


# losses = np.array(losses)
# k = 5
# worst = losses.argsort()[-k:][::-1]
# # plt.figure(figsize=(8,2))
# for i in range(k):
#     idx = worst[i]
#     print(idx)
