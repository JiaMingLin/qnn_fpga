import math
import torch
import torch.nn as nn
from torch import Tensor

from brevitas.nn import QuantLinear, QuantIdentity
from common import *

from brevitas.nn import QuantLSTM

class QuantLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True,
                 w_bit = 8, a_bit = 8, i_bit = 8, r_bit = 8):
        super(QuantLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = QuantLinear(input_size, 4 * hidden_size, weight_quant=CommonWeightQuant, 
                               bias=bias, weight_bit_width=w_bit)
        self.h2h = QuantLinear(hidden_size, 4 * hidden_size, weight_quant=CommonWeightQuant, 
                               bias=bias, weight_bit_width=w_bit)
        
        self.act_quant = QuantIdentity(act_quant=CommonActQuant, 
                                    return_quant_tensor = True, bit_width=a_bit)
        self.input_quant = QuantIdentity(act_quant=CommonActQuant, 
                                    return_quant_tensor = True, bit_width=i_bit)
        self.recurrent_quant = QuantIdentity(act_quant=CommonActQuant, 
                                    return_quant_tensor = True, bit_width=r_bit)
        
        self.c2c = Tensor(hidden_size)


        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        hx, cx = hidden

        x = x.view(-1, x.size(1))

        x = self.input_quant(x)
        
        gates = self.x2h(x) + self.h2h(hx)
        forget_gate, input_gate, cell_gate, out_gate = gates.chunk(4, 1)
        
        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        forget_gate = self.act_quant(forget_gate)
        input_gate = self.act_quant(input_gate)
        cell_gate = self.act_quant(cell_gate)
        out_gate = self.act_quant(out_gate)

        cell_gate = forget_gate * cx + input_gate * cell_gate
        hx = out_gate * self.act_quant(torch.tanh(cell_gate))

        # hx = self.recurrent_quant(hx)
        cell_gate = self.recurrent_quant(cell_gate)
        
        return hx, cell_gate

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.c2c = Tensor(hidden_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        forget_gate, input_gate, cell_gate, out_gate = gates.chunk(4, 1)
        
        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cell_gate = forget_gate * cx + input_gate * cell_gate
        hx = out_gate * torch.tanh(cell_gate)
        
        return hx, cell_gate

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bias=True, 
                 quant=False, w_bit=8, a_bit=8, i_bit=8, r_bit=8):
        super(LSTMModel, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
         
        # stack of hidden layers
        self.cells = nn.ModuleList()
        self.cells.append(
            LSTMCell(input_dim, hidden_dim, bias) if quant is False else \
            QuantLSTMCell(input_dim, hidden_dim, bias,
                          w_bit, a_bit, i_bit, r_bit)
        )
        for l in range(1,num_layers):
            self.cells.append(
                LSTMCell(hidden_dim, hidden_dim, bias) if quant is False else \
                QuantLSTMCell(hidden_dim, hidden_dim, bias,
                              w_bit, a_bit, i_bit, r_bit)
        )
    
    def forward(self, x):
        
        batch_size, seq_length, _ = x.size()
        
        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(x.device) 
             for _ in range(self.num_layers)]
        for seq in range(seq_length):
            x_t = x[:, seq, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = \
                    self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h[layer]

        out = h[-1]
        return out
    


class SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 quant = False, w_bit=8, acc_bit=16, a_bit=8, i_bit=8, o_bit=8, r_bit=8, h_bit=8,
                 no_brevitas = True):
        super(SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.quant = quant
        self.no_brevitas = no_brevitas
        if quant:
            self.rnn = QuantLSTM(
                input_size, hidden_size, num_layers, batch_first=True,
                weight_quant = weight_quantizer['int{}'.format(w_bit)],
                # io_quant = act_quantizer['int{}'.format(o_bit)],
                input_quant = act_quantizer['int{}'.format(i_bit)],
                output_quant =  NoneActQuant, # act_quantizer['int{}'.format(o_bit)],
                sigmoid_quant = act_quantizer['uint{}'.format(a_bit)],
                tanh_quant = act_quantizer['int{}'.format(a_bit)],
                hidden_state_output_quant = act_quantizer['int{}'.format(h_bit)],
                cell_state_quant = act_quantizer['int{}'.format(r_bit)],
                gate_acc_quant = act_quantizer['int{}'.format(acc_bit)]
            )
            self.quant_identity = QuantIdentity(act_quant=Int8ActPerTensorFloatScratch, 
                                    return_quant_tensor = True, bit_width=o_bit)
            self.fc = QuantLinear(hidden_size, output_size, 
                                weight_quant=weight_quantizer['int{}'.format(w_bit)]
                                )

        else:
            self.rnn = QuantLSTM(
                input_size, hidden_size, num_layers, batch_first=True,
                weight_quant = NoneWeightQuant,
                io_quant = NoneActQuant,
                gate_acc_quant = NoneActQuant,
                sigmoid_quant = NoneActQuant,
                tanh_quant = NoneActQuant,
                cell_state_quant = NoneActQuant,
                bias_quant = NoneBiasQuant
            )
            self.fc = QuantLinear(hidden_size, output_size, 
                                weight_quant=NoneWeightQuant
                             )

        if no_brevitas:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        

    def forward(self, x):
        outputs, (h_n, _) = self.rnn(x)
        outputs = outputs[:,-1,:]
        if self.quant:
            outputs = self.quant_identity(outputs)
        # outputs = self.relu(outputs)
        out = self.fc(outputs)
        return out