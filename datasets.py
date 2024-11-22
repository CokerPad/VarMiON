import numpy as np
from torch.utils import data
import torch

class SHDataset(data.Dataset):
    
    def __init__(self, theta_file, f_file, t_file, mode="train", train_sample=9000, seed = 345, index_file = "F:\Code\Steady Heat Conduction\Data\index.npy"):
        '''
        Give a example of dimension of theta, f, t
        theta -- [100, 1, 32, 32]
        f -- [100, 1, 32, 32]
        t -- [10000, 2401(49 * 49)]
        '''
        super(SHDataset, self).__init__()
        
        index = np.load(index_file)
        self.train_sample = train_sample

        self.mode = mode
        # set seed
        np.random.seed(0)

        self.theta = np.load(theta_file)
        self.f = np.load(f_file)
        self.t = np.load(t_file)

        # rearrange the temperature
        self.t = self.t[:,index]

        self.node_size = int(np.sqrt(self.t.shape[-1])) # Num of nodes in one direction

        # Create coordinates
        self.get_coordinates()


        # Random selected 100 points
        # selected_index1 = np.random.choice(np.arange(self.theta.shape[1]),size=10,replace=False)
        # selected_index2 = np.random.choice(np.arange(self.theta.shape[2]),size=10,replace=False)
        selected_index1 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
        selected_index2 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])

        theta_index = np.repeat(np.arange(self.theta.shape[0])[np.newaxis,:], 100, axis = 0).flatten()
        f_index = np.repeat(np.arange(self.f.shape[0])[np.newaxis,:], 100, axis = 1).flatten()

        if mode == "train":
            # Output should include 124 points on the boundary and distributed randomly 100 inside the domain
            index_array = np.arange(self.t.shape[-1])
            boundary_array = []

            for i in range(self.node_size):
                boundary_array.append(i)
                boundary_array.append(i * self.node_size)
                boundary_array.append(self.node_size - 1 + i * self.node_size)
                boundary_array.append(self.t.shape[-1] - 1 - i)

            self.boundary_array = np.unique(np.array(boundary_array, dtype=int))
            self.inside_array = np.setdiff1d(index_array, boundary_array)

            self.theta = self.theta[theta_index[:self.train_sample]]
            self.theta = torch.tensor(self.theta[:, selected_index1][:,:, selected_index2], dtype=torch.float32).unsqueeze(dim=1)
            self.theta = self.theta.reshape(-1, 1, 100)

            self.f = self.f[f_index[:self.train_sample]]
            self.f = torch.tensor(self.f[:, selected_index1][:,:, selected_index2], dtype=torch.float32).unsqueeze(dim=1)
            self.f = self.f.reshape(-1, 1, 100)

            self.t = self.t[:self.train_sample, :]
            self.create_output()

        else:
            self.theta = self.theta[theta_index[self.train_sample:]]
            self.theta = torch.tensor(self.theta[:, selected_index1][:,:, selected_index2], dtype=torch.float32).unsqueeze(dim=1)
            self.theta = self.theta.reshape(-1, 1, 100)

            self.f = self.f[f_index[self.train_sample:]]
            self.f = self.f = torch.tensor(self.f[:, selected_index1][:,:, selected_index2], dtype=torch.float32).unsqueeze(dim=1)
            self.f = self.f.reshape(-1, 1, 100)
            
            self.t = torch.tensor(self.t[self.train_sample:, :], dtype=torch.float32).unsqueeze(dim=1)

    def create_output(self):
            self.t_out = np.zeros((self.train_sample, 224))
            self.coordinates = np.zeros((self.train_sample, 224, 2))
            for i in range(self.train_sample):
                boundary_array = np.random.choice(self.boundary_array, size=124, replace=False)
                inside_array = np.random.choice(self.inside_array, size=100, replace=False)

                index_array = np.append(boundary_array, inside_array)
                self.t_out[i] = self.t[i, index_array]
                self.coordinates[i] = self.params[index_array]

            self.t_out = torch.tensor(self.t_out, dtype=torch.float32).unsqueeze(dim=1)
            self.coordinates = torch.tensor(self.coordinates, dtype=torch.float32)

    # def get_input(self):
        
    #     if self.mode == 'train':
    #         x = np.linspace(-1,1,self.node_size)
    #         # Attention!!! Please find out start of the element index (from -1 to 1 or from 1 to -1)
    #         y = np.linspace(-1,1,self.node_size)
    #         x, y = np.meshgrid(x, y)
    #         x, y = x.reshape(-1,1), y.reshape(-1,1)
    #         params = torch.tensor(np.concatenate((x[self.index_array % self.node_size], y[self.index_array // self.node_size]), axis=1), dtype=torch.float32).unsqueeze(dim=1)
        
    #     else:
    #         x = np.linspace(-1,1,int(np.sqrt(self.t.shape[-1])))
    #         y = np.linspace(-1,1,int(np.sqrt(self.t.shape[-1])))
    #         x, y = np.meshgrid(x, y)
    #         x, y = x.reshape(-1,1), y.reshape(-1,1)
    #         params = torch.tensor(np.concatenate((x, y), axis=1), dtype=torch.float32).unsqueeze(dim=1)
        
    #     return params

    def get_coordinates(self):
        x = np.linspace(-1,1,int(np.sqrt(self.t.shape[-1])))
        y = np.linspace(-1,1,int(np.sqrt(self.t.shape[-1])))
        x, y = np.meshgrid(x, y)
        x, y = x.reshape(-1,1), y.reshape(-1,1)
        self.params = np.concatenate((x, y), axis=1)

    def get_test_coordinates(self):

        return torch.tensor(self.params, dtype=torch.float32)


    def __getitem__(self, index):
        if self.mode == "train":
            theta = self.theta[index]
            f = self.f[index]
            label = self.t_out[index]
            coor = self.coordinates[index]
            return (theta, f, coor), label
        else:
            theta = self.theta[index]
            f = self.f[index]
            label = self.t[index]
            return (theta, f), label
        

    def __len__(self):
        return self.t.shape[0]

