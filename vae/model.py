import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1.1 : Fill in self.convs following the given architecture 
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """
        self.convs = nn.Sequential(
                    nn.Conv2d( 3,  32, kernel_size=(3,3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32,  64, kernel_size=(3,3), padding=1, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=(3,3), padding=1, stride=2),
        )
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256

        #TODO 2.1.1: fill in self.fc, such that output dimension is self.latent_dim
        self.fc = nn.Linear(self.conv_out_dim, self.latent_dim)

    def forward(self, x):
        #TODO 2.1.1 : forward pass through the network, output should be of dimension : self.latent_dim
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        #TODO 2.2.1: fill in self.fc, such that output dimension is 2*self.latent_dim
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        self.fc = nn.Linear(self.conv_out_dim, 2*latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        #TODO 2.2.1: forward pass through the network.
        # should return a tuple of 2 tensors, each of dimension self.latent_dim
        # pass
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        # print("SHould be Bx 2*1024: ", x.shape)
        return (x[:,:self.latent_dim], x[:,self.latent_dim:])

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        # TODO 2.1.1: fill in self.base_size
        # k=4, p =1, s=2
        # H =(Hin−1)*s−2*p*(k−1)+1
        self.base_size = (256, 4, 4)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        
        """
        TODO 2.1.1 : Fill in self.deconvs following the given architecture 
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.deconvs = nn.Sequential(
                    nn.ConvTranspose2d( 256, 128, kernel_size=(4,4), stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d( 128,  64, kernel_size=(4,4), padding=1, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(  64,  32, kernel_size=(4,4), padding=1, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 3, kernel_size=(3,3), padding=1, stride=1),
        )

    def forward(self, z):
        #TODO 2.1.1: forward pass through the network, first through self.fc, then self.deconvs.
        x = self.fc(z)
        x = x.reshape(-1, self.base_size[0], self.base_size[1], self.base_size[2])
        x = self.deconvs(x)
        # print("Should be (3, 32, 32): ", x.shape)
        return x

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)