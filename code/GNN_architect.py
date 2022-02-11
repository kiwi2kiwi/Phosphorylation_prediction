import dgl
import dgl.nn
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(h):
    return (h-h.mean(0))/h.std(0)

# Graph convolution layer
from dgl.nn import SGConv as ConvLayer
from dgl.nn import MaxPooling
dgl.seed(1)

layers = []
class GCN(nn.Module):

    def __init__(self, layers, kernel_size):
        super(GCN, self).__init__()
        self.convs = []
        self.n_layers = len(layers) - 1
        self.layers = layers
        # Hidden layers
        self.conv1 = ConvLayer(layers[0], layers[1], allow_zero_in_degree=True, k=kernel_size[1])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 2:
            self.conv2 = ConvLayer(layers[1], layers[2], allow_zero_in_degree=True, k=kernel_size[2])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 3:
            self.conv3 = ConvLayer(layers[2], layers[3], allow_zero_in_degree=True, k=kernel_size[3])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 4:
            self.conv4 = ConvLayer(layers[3], layers[4], allow_zero_in_degree=True, k=kernel_size[4])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 5:
            self.conv5 = ConvLayer(layers[4], layers[5], allow_zero_in_degree=True, k=kernel_size[5])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 6:
            self.conv6 = ConvLayer(layers[5], layers[6], allow_zero_in_degree=True, k=kernel_size[6])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 7:
            self.conv7 = ConvLayer(layers[6], layers[7], allow_zero_in_degree=True, k=kernel_size[7])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 8:
            self.conv8 = ConvLayer(layers[7], layers[8], allow_zero_in_degree=True, k=kernel_size[8])#, bias=False)#) #  , norm='both',
        if self.n_layers >= 9:
            self.conv9 = ConvLayer(layers[8], layers[9], allow_zero_in_degree=True, k=kernel_size[9])#, bias=False)#) #  , norm='both',

        # Output layer
        self.output = ConvLayer(layers[-1], 3, allow_zero_in_degree=True, k=kernel_size[-1])#, bias=False)#) #  , norm='both',

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        n = self.layers[1]
        if self.n_layers >= 2:
            h = self.conv2(g, h)
            h = F.relu(h)
            n = self.layers[2]
        if self.n_layers >= 3:
            h = self.conv3(g, h)
            h = F.relu(h)
            n = self.layers[3]
        if self.n_layers >= 4:
            h = self.conv4(g, h)
            h = F.relu(h)
            n = self.layers[4]
        if self.n_layers >= 5:
            h = self.conv5(g, h)
            h = F.relu(h)
            n = self.layers[5]
        if self.n_layers >= 6:
            h = self.conv6(g, h)
            h = F.relu(h)
            n = self.layers[6]
        if self.n_layers >= 7:
            h = self.conv7(g, h)
            h = F.relu(h)
            n = self.layers[7]
        if self.n_layers >= 8:
            h = self.conv8(g, h)
            h = F.relu(h)
            n = self.layers[8]
        if self.n_layers >= 9:
            h = self.conv9(g, h)
            h = F.relu(h)
            n = self.layers[9]
        h = nn.BatchNorm1d(n)(h)
        # nn.Dropout
        h = self.output(g, h)
        h = F.softmax(h, 1)
        return h