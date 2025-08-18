from sagan import *
import torchvision.models as models
from sdnet import *
import torch.nn.init as init
from causal_model import *
import copy
from torch.distributions.normal import Normal
import torch.optim as optim
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms

class ResEncoder(nn.Module):
    r'''ResNet Encoder

    Args:
        latent_dim: latent dimension
        arch: network architecture. Choices: resnet - resnet50, resnet18
        dist: encoder distribution. Choices: deterministic, gaussian, implicit
        fc_size: number of nodes in each fc layer
        noise_dim: dimension of input noise when an implicit encoder is used
    '''
    def __init__(self, latent_dim=64, arch='resnet', dist='gaussian', fc_size=2048, noise_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.dist = dist
        self.noise_dim = noise_dim

        in_channels = noise_dim + 3 if dist == 'implicit' else 3
        out_dim = latent_dim * 2 if dist == 'gaussian' else latent_dim
        if arch == 'resnet':
            self.encoder = resnet50(pretrained=False, in_channels=in_channels, fc_size=fc_size, out_dim=out_dim)
        else:
            assert arch == 'resnet18'
            self.encoder = resnet18(pretrained=False, in_channels=in_channels, fc_size=fc_size, out_dim=out_dim)

    def forward(self, x, avepool=False):
        '''
        :param x: input image
        :param avepool: whether to return the average pooling feature (used for downstream tasks)
        :return:
        '''
        if self.dist == 'implicit':
            # Concatenate noise with the input image x
            noise = x.new(x.size(0), self.noise_dim, 1, 1).normal_(0, 1)
            noise = noise.expand(x.size(0), self.noise_dim, x.size(2), x.size(3))
            x = torch.cat([x, noise], dim=1)
        z, ap = self.encoder(x)
        if avepool:
            return ap
        if self.dist == 'gaussian':
            return z.chunk(2, dim=1)
        else:
            return z


class BigDecoder(nn.Module):
    r'''Big generator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        dist: generator distribution. Choices: deterministic, gaussian, implicit
        g_std: scaling the standard deviation of the gaussian generator. Default: 1
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, dist='deterministic', g_std=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dist = dist
        self.g_std = g_std

        out_channels = 6 if dist == 'gaussian' else 3
        add_noise = True if dist == 'implicit' else False
        self.decoder = Generator(latent_dim, conv_dim, image_size, out_channels, add_noise)

    def forward(self, z, mean=False, stats=False):
        out = self.decoder(z)
        if self.dist == 'gaussian':
            x_mu, x_logvar = out.chunk(2, dim=1)
            if stats:
                return x_mu, x_logvar
            else:
                x_sample = reparameterize(x_mu, (x_logvar / 2).exp(), self.g_std)
                if mean:
                    return x_mu
                else:
                    return x_sample
        else:
            return out
class ImprovedCausalFlow(nn.Module):
    """Improved causal flow module with better structure learning"""
    def __init__(self, dim, k, C=None, hidden_dim=64, scale=True, shift=True):
        super().__init__()
        self.dim = dim  # Number of causal variables
        self.k = k      # Feature dimension per variable
        self.scale = scale
        self.shift = shift

        # Initialize causal adjacency matrix with prior knowledge if available
        if C is None:
            # Use soft structure for better gradient flow during learning
            self.C_logits = nn.Parameter(torch.zeros(dim, dim))
        else:
            # Initialize with prior knowledge but keep it learnable
            self.C_logits = nn.Parameter(torch.logit(C.float().clamp(0.1, 0.9)))
        
        # Block self-loops
        self.register_buffer('I', torch.eye(dim))
        
        # Transformation networks - more expressive with residual connections
        if scale:
            self.s_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim*k, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, k),
                    nn.Tanh()
                ) for _ in range(dim)
            ])
        
        if shift:
            self.t_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim*k, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, k)
                ) for _ in range(dim)
            ])
        
        # Adaptive flow strength with learned per-dimension parameters
        self.flow_strength = nn.Parameter(torch.ones(dim) * 0.1)
        
        # Improved sparsity control with scheduled annealing
        self.register_buffer('sparsity_weight', torch.tensor(0.1))
        self.register_buffer('temperature', torch.tensor(1.0))

        
        # Gumbel-softmax parameters for hard structure learning
        self.gumbel_temperature = 1.0
        self.use_hard_structure = False



    def get_causal_mask(self):
        """Get causal mask with improved structure learning"""
        if self.use_hard_structure:
            # Use Gumbel-softmax for hard but differentiable structure
            C_soft = F.gumbel_softmax(self.C_logits, tau=self.gumbel_temperature, hard=True)
        else:
            # Use sigmoid with temperature for soft structure
            C_soft = torch.sigmoid(self.C_logits / self.temperature)
        
        return C_soft * (1 - self.I)  # Remove self-loops

    def forward(self, z):
        """Forward pass with improved robustness"""
        batch_size = z.size(0)
        causal_mask = self.get_causal_mask()
        log_det = torch.zeros(batch_size, device=z.device)
        
        # Clone input to avoid in-place operations
        output = z.clone()

        # Process each dimension according to causal order
        for i in range(self.dim):
            # Get parent mask for this variable
            parent_mask = causal_mask[:, i].unsqueeze(1).expand(-1, self.k).reshape(-1)
            
            # Current flattened representation
            flat_z = output.reshape(batch_size, -1)
            flat=flat_z.clone()
            # Apply mask to get parent information
            masked_input = flat * parent_mask.to(z.device)
            
            # Apply scale and shift transformations
            s = self.s_nets[i](masked_input) * self.flow_strength[i] if self.scale else torch.zeros_like(z[:, i, :])
            t = self.t_nets[i](masked_input) * self.flow_strength[i] if self.shift else torch.zeros_like(z[:, i, :])
            
            # Apply transformation: y = x * exp(s) + t with gradient stabilization
            scale = torch.exp(torch.clamp(s, -5, 5))  # Clamp for stability
            output[:, i, :] = z[:, i, :] * scale + t

            log_det = log_det + torch.sum(s, dim=1)

        
        return output, log_det, causal_mask

    def inverse(self, z):
        """Inverse transformation for bidirectional modeling"""
        batch_size = z.size(0)
        causal_mask = self.get_causal_mask()
        log_det = torch.zeros(batch_size, device=z.device)
        
        # Clone input to avoid in-place operations
        output = z.clone()
        
        # Process each dimension in reverse causal order
        for i in range(self.dim-1, -1, -1):
            # Get parent mask for this variable
            parent_mask = causal_mask[:, i].unsqueeze(1).expand(-1, self.k).reshape(-1)
            
            # Current flattened representation
            flat_z = output.reshape(batch_size, -1)
            
            # Apply mask to get parent information
            masked_input = flat_z * parent_mask.to(z.device)
            
            # Apply scale and shift transformations
            s = self.s_nets[i](masked_input) * self.flow_strength[i] if self.scale else torch.zeros_like(z[:, i, :])
            t = self.t_nets[i](masked_input) * self.flow_strength[i] if self.shift else torch.zeros_like(z[:, i, :])
            
            # Apply inverse transformation: x = (y - t) / exp(s)
            scale = torch.exp(torch.clamp(s, -5, 5))  # Clamp for stability
            output[:, i, :] = (z[:, i, :] - t) / scale
            
            # Accumulate negative log determinant
            log_det = log_det - torch.sum(s, dim=1)
        
        return output, log_det, causal_mask
class BGM(nn.Module):
    """Improved Bidirectional Generative Model with better sample efficiency and robustness"""
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64,
                 enc_dist='gaussian', enc_arch='resnet', enc_fc_size=2048, enc_noise_dim=128, 
                 dec_dist='implicit', prior='gaussian', num_label=None, A=None,
                 causal_dim=4, beta_disentangle=1.0, use_vqvae=False):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.prior_dist = prior
        self.num_label = num_label
        self.use_vqvae = use_vqvae

        # Encoder with improved regularization
        self.encoder = ResEncoder(latent_dim, enc_arch, enc_dist, enc_fc_size, enc_noise_dim)
        # Apply spectral normalization to encoder for better stability
        self.apply_spectral_norm(self.encoder)
        
        # Decoder 
        self.decoder = BigDecoder(latent_dim, conv_dim, image_size, dec_dist)
        
        # Disentanglement parameters
        self.beta_disentangle = beta_disentangle
        self.causal_dim = causal_dim
        
        # Feature dimension per causal variable
        k = latent_dim // causal_dim
        
        # Prior model
        if 'scm' in prior:
            self.prior = SCM(num_label, A, scm_type=prior)
        
        # Create causal adjacency matrix
        if A is None and num_label is not None:
            C = torch.zeros(causal_dim, causal_dim)
            # Set some basic causal relationships as initialization
            for i in range(min(num_label-1, causal_dim-1)):
                C[i, i+1:min(i+3, causal_dim)] = 1.0
        else:
            # Extend existing causal relationships to all dimensions
            C = torch.zeros(causal_dim, causal_dim)
            if A is not None:
                C[:A.size(0), :A.size(1)] = A
        
        # Improved causal flow module
        self.causal_flow = ImprovedCausalFlow(dim=causal_dim, k=k, C=C)
        

    
    def encode(self, x, mean=False, avepool=False):
        """Enhanced encoding function with causal flow processing"""
        if avepool:
            return self.encoder(x, avepool=True)
        else:
            if self.enc_dist == 'gaussian':
                # Get encoder output
                z_mu, z_logvar = self.encoder(x)
                

                if mean:  # For downstream tasks
                    return z_mu
                else:
                    # Reparameterize using transformed z
                    z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
                    return z_fake
            else:
                return self.encoder(x)
    
    def forward(self, x=None, z=None, recon=False):

        # Training mode
        if x is not None and z is not None:
            if self.enc_dist == 'gaussian':
                # Get encoder output
                z_mu, z_logvar = self.encoder(x)
                k = self.latent_dim // self.causal_dim
                z_shaped = z_mu.reshape([z_mu.size(0), self.causal_dim, k])
                # Apply causal flow
                z_flow, log_det, causal_mask = self.causal_flow(z_shaped)
                # Reshape back to original format
                z_flow_flat = z_flow.reshape([z_mu.size(0), self.latent_dim])
                # Apply post-processing normalization
                z_final = z_flow_flat.reshape([z_mu.size(0), self.causal_dim, k])
                # Original reparameterized sample
                z_fake = reparameterize(z_final, (z_logvar / 2).exp())

            # Handle prior (if using SCM)
            if 'scm' in self.prior_dist:
                label_z = self.prior(z[:, :self.num_label])
                other_z = z[:, self.num_label:]
                z = torch.cat([label_z, other_z], dim=1)

            # Generate reconstruction
            x_fake = self.decoder(z)
            if recon:
                return self.decoder(z_fake)

        return z_fake, x_fake, z, z_final,log_det

            



class BigJointDiscriminator(nn.Module):
    r'''Big joint discriminator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        fc_size: number of nodes in each fc layers
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, fc_size=1024):
        super().__init__()
        self.discriminator = Discriminator(conv_dim, image_size, in_channels=3, out_feature=True)
        self.discriminator_z = Discriminator_MLP(latent_dim, fc_size)
        self.discriminator_j = Discriminator_MLP(conv_dim * 16 + fc_size, fc_size)

    def forward(self, x, z):
        sx, feature_x = self.discriminator(x)
        sz, feature_z = self.discriminator_z(z)
        sxz, _ = self.discriminator_j(torch.cat((feature_x, feature_z), dim=1))
        return (sx + sz + sxz) / 3
def reparameterize(mu, sigma, std=1):
    assert mu.shape == sigma.shape
    eps = mu.new(mu.shape).normal_(0, std)
    return mu + sigma * eps

def kl_div(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

def gaussian_nll(x_mu, x_logvar, x):
    '''NLL'''
    sigma_inv = (- x_logvar / 2).exp()
    return 0.5 * (x_logvar + ((x - x_mu) * sigma_inv).pow(2) + np.log(2*np.pi)).sum()

def kaiming_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # kaiming_uniform_(m.weight)
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.BatchNorm1d or type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)