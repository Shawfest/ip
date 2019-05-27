import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, layersize, eta=None):
        super(NN, self).__init__()

    def forward(self, x):
        return x
    
    def update(self, u, v, eta=None):
        pass
    

    
# Master BN module
class BN(nn.Module):
    def __init__(self, layersize, eta=1):
        super(BN, self).__init__()
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        # Alpha and beta are the ip normalization parameters
        self.register_buffer('alpha', torch.ones(layersize))
        self.register_buffer('beta', torch.zeros(layersize))
        
        self.alpha = None
        self.beta = None
        
        self.eta = eta
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return o

    def update(self, u, v, eta=None):
        
        if (eta is None):
            eta = self.eta
            
        with torch.no_grad():
            beta = u.mean(0, keepdim=True)
            alpha = ((u-beta)**2).mean(0, keepdim=True).sqrt()
        
        self.alpha = (1-eta)*self.alpha + eta * alpha
        self.beta = (1-eta)*self.beta + eta * beta
    

# IP Normalization layers
class IP(nn.Module):
    def __init__(self, layersize, eta=1):
        super(IP, self).__init__()
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        # Alpha and beta are the ip normalization parameters
        self.register_buffer('alpha', torch.ones(layersize))
        self.register_buffer('beta', torch.zeros(layersize))
        
        self.eta = eta
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return o

    def update(self, u, v, eta=None):
        
        if (eta is None):
            eta = self.eta
            
        with torch.no_grad():
            beta = u.median(0, keepdim=True)
            muU = u.sum(0, keepdim=True)/(u.shape[0]-1)
            muV = v.sum(0, keepdim=True)/(u.shape[0]-1)
            alpha = ((u-muU)*(v-muV)).mean(0, keepdim=True)
        
        self.alpha = (1-eta)*self.alpha + eta * alpha
        self.beta = (1-eta)*self.beta + eta * beta[0]
        
# Median-Variance normalization
class DV(nn.Module):
    def __init__(self, layersize, eta=None):
        super(BN, self).__init__()
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        # Alpha and beta are the ip normalization parameters
        self.register_buffer('alpha', torch.ones(layersize))
        self.register_buffer('beta', torch.zeros(layersize))
        
        self.alpha = None
        self.beta = None
        
        self.eta = eta
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return o

    def update(self, u, v, eta=None):
        
        if (eta is None):
            eta = self.eta
            
        with torch.no_grad():
            beta = u.median(0, keepdim=True)
            muU = u.mean(0, keepdim=True)
            alpha = ((u-muU)**2).mean(0, keepdim=True).sqrt()
        
        self.alpha = (1-eta)*self.alpha + eta * alpha
        self.beta = (1-eta)*self.beta + eta * beta[0]