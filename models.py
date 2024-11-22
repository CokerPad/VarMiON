import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

class FullyConnectedNN(nn.Module):
    def __init__(self, n_neurons, num_layer, input_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, n_neurons)
        # self.input_bn = nn.BatchNorm1d(1)
        self.output_layer = nn.Linear(n_neurons, output_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
            # self.layers.append(nn.BatchNorm1d(1))

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        # for i in range(self.n_layers-1):
        #     x = self.layers[i](x)
        #     if i < self.n_layers-2:
        #         x = F.relu(x)

        # for i in range(len(self.layers)):
        #     if i % 2 == 0:
        #         x = F.relu(self.layers[i](x))
        #     else:
        #         x = self.layers[i](x)

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.output_layer(x)
        

        return x
    
class Branch_Net(nn.Module):
    def __init__(self):
        super(Branch_Net, self).__init__()
        self.Linear1 = nn.Linear(100, 100)
        self.Linear2 = nn.Linear(100, 512)
        self.LinearF1 = nn.Linear(100, 128)
        self.Fbn1 = nn.BatchNorm1d(1)
        self.LinearF2 = nn.Linear(128,256)
        self.Fbn2 = nn.BatchNorm1d(1)
        self.LinearF3 = nn.Linear(256, 128)
        self.Fbn3 = nn.BatchNorm1d(1)
        self.LinearF4 = nn.Linear(128,64)

        self.TrConv1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.bn1 = nn.BatchNorm2d(16)

        self.TrConv2 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.bn2 = nn.BatchNorm2d(16)

        self.TrConv3 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.bn3 = nn.BatchNorm2d(8)

        self.TrConv4 = nn.ConvTranspose2d(8, 1, 2, 2)


    
    def forward(self, theta, f):

        theta = F.relu(self.Linear1(theta))
        theta = F.relu(self.Linear2(theta))

        # Reshape theta
        theta = theta.reshape(-1, 32, 4, 4)

        theta = self.bn1(F.relu(self.TrConv1(theta)))
        theta = self.bn2(F.relu(self.TrConv2(theta)))
        theta = self.bn3(F.relu(self.TrConv3(theta)))

        theta = F.tanhshrink(self.TrConv4(theta))

        f = self.Fbn1(F.relu(self.LinearF1(f)))
        f = self.Fbn2(F.relu(self.LinearF2(f)))
        f = self.Fbn3(F.relu(self.LinearF3(f)))
        f = self.LinearF4(f)

        f = f.unsqueeze(dim=-1)
        
        output = torch.matmul(theta, f).sum(dim=-1)

        return output
    
class Mini_Branch_Net(nn.Module):
    def __init__(self):
        super(Mini_Branch_Net, self).__init__()
        self.Linear1 = nn.Linear(100, 100)
        self.Linear2 = nn.Linear(100, 512)

        self.LinearF1 = nn.Linear(100, 64)
        # self.Fbn1 = nn.BatchNorm1d(1)
        # self.LinearF2 = nn.Linear(128,256)
        # self.Fbn2 = nn.BatchNorm1d(1)
        # self.LinearF3 = nn.Linear(256, 128)
        # self.Fbn3 = nn.BatchNorm1d(1)
        # self.LinearF4 = nn.Linear(128,64)

        self.TrConv1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.bn1 = nn.BatchNorm2d(16)

        self.TrConv2 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.bn2 = nn.BatchNorm2d(16)

        self.TrConv3 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.bn3 = nn.BatchNorm2d(8)

        self.TrConv4 = nn.ConvTranspose2d(8, 1, 2, 2)


    
    def forward(self, theta, f):

        theta = F.relu(self.Linear1(theta))
        theta = F.relu(self.Linear2(theta))

        # Reshape theta
        theta = theta.reshape(-1, 32, 4, 4)

        theta = self.bn1(F.relu(self.TrConv1(theta)))
        theta = self.bn2(F.relu(self.TrConv2(theta)))
        theta = self.bn3(F.relu(self.TrConv3(theta)))

        theta = F.tanhshrink(self.TrConv4(theta))

        # f = self.Fbn1(F.relu(self.LinearF1(f)))
        # f = self.Fbn2(F.relu(self.LinearF2(f)))
        # f = self.Fbn3(F.relu(self.LinearF3(f)))
        f = self.LinearF1(f)

        f = f.unsqueeze(dim=-1)
        
        output = torch.matmul(theta, f).sum(dim=-1)

        return output

class VarMiON(nn.Module):
    
    def __init__(self, trunk_net, branch_net):
        super(VarMiON, self).__init__()
        self.Trunk = trunk_net
        self.Branch = branch_net

    def forward(self, theta, f, x):

        branch_out = self.Branch(theta, f)
        trunk_out = self.Trunk(x)
        # dot_product = torch.zeros((branch_out.shape[0], 1, trunk_out.shape[0]), device=device)
        # for i in range(trunk_out.shape[0]):
        #     print(branch_out.shape)
        #     print(trunk_out[i].unsqueeze(dim=-1).shape)
        #     dot_product[:,:, i] = torch.matmul(branch_out, trunk_out[i].unsqueeze(dim=-1))

        dot_product = (branch_out * trunk_out).sum(-1).unsqueeze(dim=1)


        return dot_product


