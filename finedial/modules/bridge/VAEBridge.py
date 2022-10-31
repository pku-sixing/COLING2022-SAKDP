import torch
from torch import nn

class VAEBridge(nn.Module):

    def __init__(self, input_dim, output_dim):
        """
            @parameter:
                input_dim : 输入的维度大小
                output_dim : 输出的隐维度大小
        """
        super().__init__()
        self.mu_projection = nn.Linear(input_dim, output_dim)
        self.var_projection = nn.Linear(input_dim, output_dim)

    def reparameterize(self, input):
        mu = self.mu_projection(input)
        log_var = self.var_projection(input)

        std = torch.exp(log_var / 2.0)
        eps = torch.randn_like(std)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return mu + eps * std, kl_div

    def forward(self, input):
        return self.reparameterize(input)

