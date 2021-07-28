import torch
from torch import Tensor


def reparameterize(
        mu: Tensor,
        logvar: Tensor) -> Tensor:
    """
    Reparameterize trick
    :param mu: mean, torch.float32, (batch_size, dim_latent)
    :param logvar: log(var), torch.float32, (batch_size, dim_latent)
    :return: reparameterized latent variable, torch.float32, (batch_size, dim_latent)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def batch_reparameterize(mu, logvar, num_samples: int):
    """
    Reparameterize trick in a batch
    :param mu: mean, torch.float32, (batch_size, dim_latent)
    :param logvar: log(var), torch.float32, (batch_size, dim_latent)
    :param num_samples: number of target samples
    :return: reparameterized latent variable, torch.float32, (batch_size, num_samples, dim_latent)
    """
    std = torch.exp(0.5 * logvar)  # (batch_size, dim_latent)
    std = std.unsqueeze(1).expand(-1, num_samples, -1)  # (batch_size, num_samples, dim_latent)
    mu = mu.unsqueeze(1).expand(-1, num_samples, -1)  # (batch_size, num_samples, dim_latent)
    eps = torch.randn_like(std)
    return mu + eps * std


if __name__ == '__main__':
    m = torch.randn(10, 20)
    v = torch.randn(10, 20)
    z = batch_reparameterize(m, v, 17)
    print(z.shape)
