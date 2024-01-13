import torch.nn as nn
import torch
class ShiftBatchNorm(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()

        self.epsilon = 1E-5     # Same as PyTorch

        # Trainable parameters
        self.beta = torch.nn.Parameter(data=torch.tensor(0.0))
        self.gamma = torch.nn.Parameter(data=torch.tensor(1.0))
        self.register_parameter('beta', self.beta)
        self.register_parameter('gamma', self.gamma)
        self.running_mean = 0
        self.running_var = 0
        self.momentum = momentum

    def forward(self, x):
        batch_mean = torch.mean(x)
        batch_var = torch.var(x, unbiased=False)
        
        def ap2(x):
            return torch.sign(x) * (2**(torch.round(torch.log2(torch.abs(x)))))
        
        self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * batch_mean
        self.running_var = self.momentum * self.running_var + (1-self.momentum) * batch_var
        
        x_hat = (x - batch_mean) / ap2(torch.sqrt(batch_var + self.epsilon))
#         x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.epsilon)
        result = x_hat * ap2(self.gamma) + self.beta

        return result