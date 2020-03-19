'''
Modified by Honey Gupta. The original scripts can be found at https://github.com/erwulff/lth_thesis_project and https://github.com/Skelpdar/HEPAutoencoders.

Functions were modified or added for 4D data. The ones related to 27D AOD data were removed for better readibility.
'''

import time
import torch
import datetime
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

class AE_bn_LeakyReLU(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_bn_LeakyReLU, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))
            en_modulelist.append(nn.LeakyReLU())
            en_modulelist.append(nn.BatchNorm1d(outs))
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))
            de_modulelist.append(nn.LeakyReLU())
            de_modulelist.append(nn.BatchNorm1d(outs))

        de_modulelist = de_modulelist[:-2]  # Remove LeakyReLU activation and BatchNorm1d from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)
        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string

class AE_bn_ELU(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_bn_ELU, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))
            en_modulelist.append(nn.ELU())
            en_modulelist.append(nn.BatchNorm1d(outs))
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))
            de_modulelist.append(nn.ELU())
            de_modulelist.append(nn.BatchNorm1d(outs))

        de_modulelist = de_modulelist[:-2]  # Remove LeakyReLU activation and BatchNorm1d from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)
        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string

class AE_3D_200(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        m1 = self.en1(x)
        m2 = self.tanh(m1)
        m3 = self.en2(m2)
        m4 = self.tanh(m3)
        m5 = self.en3(m4)
        m6 = self.tanh(m5)
        return self.en4(m6)

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'

class AE_3D_200_no_tanh(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200_no_tanh, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        m1 = self.en1(x)
        m2 = self.en2(m1)
        m3 = self.en3(m2)
        return self.en4(m3)

    def decode(self, x):
        return self.de4(self.de3(self.de2(self.de1(x))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'

class AE_big_2D_v1(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_2D_v1, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 2)
        self.de1 = nn.Linear(2, 4)
        self.de2 = nn.Linear(4, 6)
        self.de3 = nn.Linear(6, 8)
        self.de4 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-2-4-6-8-out'

class AE_big_2D_v2(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_2D_v2, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 3)
        self.en5 = nn.Linear(3, 2)
        self.de1 = nn.Linear(2, 3)
        self.de2 = nn.Linear(3, 4)
        self.de3 = nn.Linear(4, 6)
        self.de4 = nn.Linear(6, 8)
        self.de5 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en5(self.tanh(self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))))

    def decode(self, x):
        return self.de5(self.tanh(self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-3-2-3-4-6-8-out'

class AE_2D(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D, self).__init__()
        self.en1 = nn.Linear(n_features, 20)
        self.en2 = nn.Linear(20, 10)
        self.en3 = nn.Linear(10, 6)
        self.en4 = nn.Linear(6, 2)
        self.de1 = nn.Linear(2, 6)
        self.de2 = nn.Linear(6, 10)
        self.de3 = nn.Linear(10, 20)
        self.de4 = nn.Linear(20, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-20-10-6-2-6-10-20-out'

class AE_2D_v2(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v2, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 20)
        self.en3 = nn.Linear(20, 10)
        self.en4 = nn.Linear(10, 2)
        self.de1 = nn.Linear(2, 10)
        self.de2 = nn.Linear(10, 20)
        self.de3 = nn.Linear(20, 50)
        self.de4 = nn.Linear(50, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-20-10-2-10-20-50-out'

class AE_big_2D_v3(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_2D_v3, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 2)
        self.de1 = nn.Linear(2, 6)
        self.de2 = nn.Linear(6, 8)
        self.de3 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))

    def decode(self, x):
        return self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-2-6-8-out'

class AE_2D_v3(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v3, self).__init__()
        self.en1 = nn.Linear(n_features, 100)
        self.en2 = nn.Linear(100, 200)
        self.en3 = nn.Linear(200, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 200)
        self.de3 = nn.Linear(200, 100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-100-200-100-2-100-200-100-out'

class AE_2D_v4(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v4, self).__init__()
        self.en1 = nn.Linear(n_features, 500)
        self.en2 = nn.Linear(500, 200)
        self.en3 = nn.Linear(200, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 200)
        self.de3 = nn.Linear(200, 500)
        self.de4 = nn.Linear(500, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-500-200-100-2-100-200-500-out'

class AE_2D_v5(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v5, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 2)
        self.de1 = nn.Linear(2, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-2-50-100-200-out'

class AE_2D_v100(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v100, self).__init__()
        self.en1 = nn.Linear(n_features, 100)
        self.en2 = nn.Linear(100, 100)
        self.en3 = nn.Linear(100, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 100)
        self.de3 = nn.Linear(100, 100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-100-100-100-2-100-100-100-out'

class AE_2D_v50(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v50, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 50)
        self.en3 = nn.Linear(50, 50)
        self.en4 = nn.Linear(50, 2)
        self.de1 = nn.Linear(2, 50)
        self.de2 = nn.Linear(50, 50)
        self.de3 = nn.Linear(50, 50)
        self.de4 = nn.Linear(50, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-50-2-50-50-50-out'


class AE_2D_v1000(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v1000, self).__init__()
        self.en1 = nn.Linear(n_features, 1000)
        self.en2 = nn.Linear(1000, 400)
        self.en3 = nn.Linear(400, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 400)
        self.de3 = nn.Linear(400, 1000)
        self.de4 = nn.Linear(1000, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-1000-400-100-2-100-400-1000-out'
