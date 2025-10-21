import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

def th2np(tensor):
    """Utility to convert a PyTorch tensor to a NumPy array"""
    return tensor.detach().cpu().numpy()

def get_argmax_mu(pi, mu):
    """
    Get the mean of the most probable Gaussian mixture component.
    :param pi: [N x K x D] (Mixture probabilities)
    :param mu: [N x K xD] (Mixture means)
    :return: [N x D] (The most probable mean)
    """
    max_idx = th.argmax(pi, dim=1) # [N x D]
    argmax_mu = th.gather(input=mu, dim=1, index=max_idx.unsqueeze(dim=1)).squeeze(dim=1) # [N x D]
    return argmax_mu

def gmm_forward(pi, mu, sigma, data):
    """
    Compute Gaussian mixture model probability and NLL loss.
    
    :param pi: GMM mixture weights [N x K x D]
    :param mu: GMM means [N x K x D]
    :param sigma: GMM stds [N x K x D]
    :param data: data [N x D]
    :return: Dictionary containing probabilities and NLLs
    """
    data_usq = th.unsqueeze(data, 1) # [N x 1 x D]
    data_exp = data_usq.expand_as(sigma) # [N x K x D]
    ONEOVERSQRT2PI = 1 / np.sqrt(2 * np.pi)
    
    # Calculate the probability of data belonging to each Gaussian component
    probs = ONEOVERSQRT2PI * th.exp(-0.5 * ((data_exp - mu) / sigma)**2) / sigma # [N x K x D]
    
    # Weight probabilities by mixture weights (pi) and sum them up
    probs = probs * pi # [N x K x D]
    probs = th.sum(probs, dim=1) # [N x D]
    
    # To prevent numerical instability (log(0))
    probs = th.clamp(probs, min=1e-8) 
    
    # Calculate log-probability and Negative Log-Likelihood (NLL)
    log_probs = th.log(probs) # [N x D]
    log_probs = th.sum(log_probs, dim=1) # [N]
    nlls = -log_probs # [N]

    # Most probable mean [N x D]
    argmax_mu = get_argmax_mu(pi, mu) # [N x D]
    
    out = {
        'probs': probs, 'log_probs': log_probs, 'nlls': nlls, 'argmax_mu': argmax_mu
    }
    return out

class MixturesOfGaussianLayer(nn.Module):
    """The final layer of the MDN, outputting pi, mu, and sigma."""
    def __init__(
        self,
        in_dim,
        out_dim,
        k,
        sig_max=None
    ):
        super(MixturesOfGaussianLayer, self).__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.k       = k
        self.sig_max = sig_max
        
        # Networks
        self.fc_pi    = nn.Linear(self.in_dim, self.k * self.out_dim)
        self.fc_mu    = nn.Linear(self.in_dim, self.k * self.out_dim)
        self.fc_sigma = nn.Linear(self.in_dim, self.k * self.out_dim)

    def forward(self, x):
        pi_logit = self.fc_pi(x) # [N x KD]
        pi_logit = th.reshape(pi_logit, (-1, self.k, self.out_dim)) # [N x K x D]
        pi       = th.softmax(pi_logit, dim=1) # [N x K x D]
        
        mu       = self.fc_mu(x) # [N x KD]
        mu       = th.reshape(mu, (-1, self.k, self.out_dim)) # [N x K x D]
        
        sigma    = self.fc_sigma(x) # [N x KD]
        sigma    = th.reshape(sigma, (-1, self.k, self.out_dim)) # [N x K x D]
        
        if self.sig_max is None:
            sigma = th.exp(sigma) # [N x K x D]
        else:
            sigma = self.sig_max * th.sigmoid(sigma) # [N x K x D]
        
        # Clamp sigma to avoid division by zero
        sigma = th.clamp(sigma, min=1e-5)
            
        return pi, mu, sigma

class MixtureDensityNetwork(nn.Module):
    """The full MDN model."""
    def __init__(
        self,
        name       = 'mdn',
        x_dim      = 1,
        y_dim      = 1,
        k          = 5,
        h_dim_list = [32, 32],
        actv       = nn.ReLU(),
        sig_max    = 1.0,
        mu_min     = -3.0,
        mu_max     = +3.0,
        p_drop     = 0.1,
        use_bn     = False,
    ):
        super(MixtureDensityNetwork, self).__init__()
        self.name       = name
        self.x_dim      = x_dim
        self.y_dim      = y_dim
        self.k          = k
        self.h_dim_list = h_dim_list
        self.actv       = actv
        self.sig_max    = sig_max
        self.mu_min     = mu_min
        self.mu_max     = mu_max
        self.p_drop     = p_drop
        self.use_bn     = use_bn

        # Declare layers
        self.layer_list = []
        h_dim_prev = self.x_dim
        for h_dim in self.h_dim_list:
            # dense -> batchnorm -> actv -> dropout
            self.layer_list.append(nn.Linear(h_dim_prev, h_dim))
            if self.use_bn: self.layer_list.append(nn.BatchNorm1d(num_features=h_dim))
            self.layer_list.append(self.actv)
            if self.p_drop > 0: self.layer_list.append(nn.Dropout(p=self.p_drop)) # Use nn.Dropout for 1D data
            h_dim_prev = h_dim
        
        self.layer_list.append(
            MixturesOfGaussianLayer(
                in_dim = h_dim_prev,
                out_dim = self.y_dim,
                k = self.k,
                sig_max = self.sig_max
            )
        )

        # Define network
        self.net = nn.Sequential()
        self.layer_names = []
        for l_idx, layer in enumerate(self.layer_list):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.layer_names.append(layer_name)
            self.net.add_module(layer_name, layer)

        # Initialize parameters
        self.init_param(VERBOSE=False)

    def init_param(self, VERBOSE=False):
        """Initialize parameters"""
        for m_idx, m in enumerate(self.modules()):
            if VERBOSE: print ("[%02d]" % (m_idx))
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # (Heuristics) mu bias between mu_min ~ mu_max
        self.layer_list[-1].fc_mu.bias.data.uniform_(self.mu_min, self.mu_max)
        
    def forward(self, x):
        """Forward propagate"""
        return self.net(x)