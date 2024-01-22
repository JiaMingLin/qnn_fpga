import torch.nn as nn
import torch
class ShiftBatchNorm(nn.Module):
    def __init__(self, size, momentum=0.1):
        super().__init__()

        self.epsilon = 1E-5     # Same as PyTorch

        # Trainable parameters
        self.size = size
        self.register_parameter('beta', torch.nn.Parameter(data=torch.zeros(size)))
        self.register_parameter('gamma', torch.nn.Parameter(data=torch.ones(size)))
        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones(size))
        self.momentum = momentum

    def forward(self, x):

        def ap2(x):
            return torch.sign(x) * (2**(torch.round(torch.log2(torch.abs(x)))))

        result = None
        if self.training: # training
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, unbiased=False, dim=0)

            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * batch_var

            x_hat = (x - batch_mean) / ap2(torch.sqrt(batch_var + self.epsilon))
        
            result = x_hat * ap2(self.gamma) + self.beta
        else: # inferencing
            x_hat = (x - self.running_mean) / ap2(torch.sqrt(self.running_var + self.epsilon))
            result = x_hat * ap2(self.gamma) + self.beta

        return result