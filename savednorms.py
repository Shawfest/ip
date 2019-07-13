class BN_H(nn.Module):
    def __init__(self, layersize, eta=None):
        super(BN_H, self).__init__()
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        self.alpha = None
        self.beta = None
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return o

    def update(self, u, v, eta=None):
        
        with torch.no_grad():
            beta = u.mean(0, keepdim=True)
            alpha = ((u-beta)**2).mean(0, keepdim=True).sqrt()
        
        self.alpha = alpha
        self.beta = beta
    

class BN_I(nn.Module):
    def __init__(self, layersize, eta=1):
        super(BN_I, self).__init__()
        self.eta = eta
        
        # gain/bias are the learned output distribution params
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        # Alpha and beta are the ip normalization parameters
        self.register_buffer('alpha', torch.ones(layersize))
        self.register_buffer('beta', torch.zeros(layersize))
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return  o
        
    def update(self, u, v, eta=None):

        if (eta is None):
            eta = self.eta
        
        with torch.no_grad():
            Eu = u.mean(0, keepdim=True)
            Euu = (u**2).mean(0, keepdim=True)
            Ev = v.mean(0, keepdim=True)
            Evv = (v**2).mean(0, keepdim=True)
            Euv = (u*v).mean(0, keepdim=True)

        self.alpha = (1-eta)*self.alpha + eta * ((Euu - Eu**2))
        self.beta = (1-eta)*self.beta + eta * (Eu)

        
        
# IP Normalization layers
class IP_H(nn.Module):
    def __init__(self, layersize, eta=None):
        super(IP_H, self).__init__()
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        self.alpha = None
        self.beta = None
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return o

    def update(self, u, v, eta=None):
        
        with torch.no_grad():
            beta = u.median(0, keepdim=True)
            muU = u.sum(0, keepdim=True)/(u.shape[0]-1)
            muV = v.sum(0, keepdim=True)/(u.shape[0]-1)
            alpha = ((u-muU)*(v-muV)).mean(0, keepdim=True)
        
        self.alpha = alpha
        self.beta = beta
    

class IP_I(nn.Module):
    def __init__(self, layersize, eta=1):
        super(IP_I, self).__init__()
        self.eta = eta
        
        # gain/bias are the learned output distribution params
        self.gain = nn.Parameter(torch.ones(layersize))
        self.bias = nn.Parameter(torch.zeros(layersize))
        
        # Alpha and beta are the ip normalization parameters
        self.register_buffer('alpha', torch.ones(layersize))
        self.register_buffer('beta', torch.zeros(layersize))
        
    def forward(self, x):

        # Normalize
        nx = (x-self.beta)/self.alpha

        # Adjust using learned parameters
        o = self.gain*nx + self.bias
        return  o
        
    def update(self, u, v, eta=None):

        if (eta is None):
            eta = self.eta
        
        with torch.no_grad():
            Eu = u.mean(0, keepdim=True)
            Euu = (u**2).mean(0, keepdim=True)
            Ev = v.mean(0, keepdim=True)
            Evv = (v**2).mean(0, keepdim=True)
            Euv = (u*v).mean(0, keepdim=True)

        self.alpha = (1-eta)*self.alpha + eta * ((Euv - Eu*Ev))
        self.beta = (1-eta)*self.beta + eta * (Ev)
        