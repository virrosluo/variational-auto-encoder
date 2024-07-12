import torch
from torch import nn
from torch.nn import functional as F

from model.resnet_model import (
    resnet18_encoder,
    resnet18_decoder
)

class VAE(nn.Module):
    def __init__(
        self, 
        enc_out_dim=512, 
        latent_dim=256,
        input_height=32
    ):
        super().__init__()
        
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # mean, log_variance mapping function
        self.mu = nn.Linear(enc_out_dim, latent_dim)
        self.log_var = nn.Linear(enc_out_dim, latent_dim)

        # Gaussian likelihood mapping from decoder output
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def sampling_z_value(self, mu: torch.Tensor, log_var: torch.Tensor):
        '''
            Sampling Z value from q(Z | X) ~ Normal(Z, (mu(X), var(X)))
        '''

        std = torch.exp(log_var)

        #
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z

    def kl_divergence_loss(
        self, 
        z: torch.Tensor, 
        mu: torch.Tensor, 
        log_var: torch.Tensor):
        '''
            Using Monte Carlo KL Divergence to approximating the analytic solving the equation to become computable function.
            This can be calculate by using the mu and std getting from encoding value x, the sampled z and calculate the difference between current distribution and prior distribution

            `z`: The sampled vectors from current optimizing distribution q
            `mu`: The mean that extracted from input image `self.mu(encoded_x)`
            `log_var`: The log variance that extracted from input image `self.log_var(encoded_x)
        '''

        # Prior distribution is multivariate normal distribution with mu = 0 and std = 1
        std = torch.exp(log_var)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        
        # Current optimizing distribution
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z) # Shape: Batch_size
        log_pz = p.log_prob(z) # Shape: Batch_size

        kl = log_qzx - log_pz
        return kl.sum(-1)

    def reconstruct_loss(
        self, 
        mean: torch.Tensor, 
        log_scale: torch.Tensor, 
        sample: torch.Tensor):
        '''
            Computing reconstructing origin image from mean and log_var which extracted from sampled `z` values
            Sometimes, we can easily calculate the reconstruct_loss by calculating the difference between generated_img and original_img by MSE. But this only true when we assume the p(X | Z) ~ Normal
            `mean`: The extracted mean getting from decoding `z_value`
            `log_scale`: The extracted mean getting from decoding `z_value`
            `sample`: The original image that we need to calculate the probability appearing given `z_value`
            
        '''
        scale = torch.exp(log_scale)
        dist = torch.distributions.Normal(mean, scale)

        log_pxz = dist.log_prob(sample)
        return log_pxz.sum()
