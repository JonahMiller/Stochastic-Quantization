import math
from functools import partial

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class ELQScheduler(object):
    def __init__(self, optimizer, iterative_steps, strategy="pruning"):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if strategy not in "pruning":
            raise ValueError("ELQ supports \"pruning\" -inspired weight partitioning")
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.strategy = strategy
        self.idx = 0

        for group in self.optimizer.param_groups:
            group['alpha'] = []
            group['Ts'] = []
            group['diff'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['Ts'].append(0)
                    continue
                T = torch.ones_like(p.data).cuda()
                group['Ts'].append(T)
                alpha = torch.mean(p.data.abs()) + 0.05*torch.max(p.data.abs())
                group['alpha'].append(alpha)
                group['diff'].append(0)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def quantize(self):
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                T = group['Ts'][idx]
                alpha = group['alpha'][idx]
                fully_quantized = self.quantize_weight(p.data, alpha)
                p.data = torch.where(T == 0, fully_quantized, p.data)
                group['diff'][idx] = torch.sign(p.data - fully_quantized)

    def quantize_weight(self, tensor, alpha):
        a = (tensor > 0.5*alpha).float()
        b = (tensor < -0.5*alpha).float()
        return alpha*(a - b)

    def step(self):
        for group in self.optimizer.param_groups:
            print(f"Step: {self.iterative_steps[self.idx]}")
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue

                old_T = group['Ts'][idx]

                new_T = self.weight_bounds_T(p.data, 0.5,
                                             self.iterative_steps[self.idx],
                                             group['alpha'][idx])
                weight_clip_T = self.excess_weight_bounds_T(p.data, group['alpha'][idx])
                temp_T = torch.mul(new_T, old_T)
                T = torch.mul(temp_T, weight_clip_T)
                group['Ts'][idx] = T

        self.quantize()
        self.idx += 1

    def finish_quantize(self):
        for group in self.optimizer.param_groups:
            print(f"Step: ALL QUANTISED")
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue

                group['Ts'][idx] = torch.zeros_like(p.data)
        self.quantize()

    def weight_bounds_T(self, tensor, sigma1, sigma2, alpha):
        minr, maxr = sigma2*alpha, (2*sigma1 - sigma2)*alpha
        abs_tensor = tensor.abs()
        zeros = torch.zeros_like(tensor)
        ones = torch.ones_like(tensor)
        T1 = torch.where(abs_tensor < maxr, ones, zeros)
        T2 = torch.where(minr <= abs_tensor, ones, zeros)
        and_gate = torch.mul(T1, T2)
        nor_gate = ones - and_gate
        return nor_gate

    def excess_weight_bounds_T(self, tensor, alpha):
        abs_tensor = tensor.abs()
        zeros = torch.zeros_like(tensor)
        ones = torch.ones_like(tensor)
        T = torch.where(abs_tensor > 2*alpha, ones, zeros)
        nor_gate = ones - T
        return nor_gate


class SQ_ELQScheduler_custom_layer(object):
    def __init__(self, optimizer, r_steps, prob_type, e_type, strategy="pruning"):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if strategy not in "pruning":
            raise ValueError("ELQ supports \"pruning\" -inspired weight partitioning")
        self.optimizer = optimizer
        self.r_steps = r_steps
        self.strategy = strategy
        self.idx = 0
        self.prob_type = prob_type
        self.e_type = e_type

        for group in self.optimizer.param_groups:
            group['alpha'] = []
            group['Ts'] = []
            group['diff'] = []
            group['error'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['Ts'].append(0)
                    continue
                T = torch.ones_like(p.data).cuda()
                group['Ts'].append(T)
                alpha = torch.mean(p.data.abs()) + 0.05*torch.max(p.data.abs())
                group['alpha'].append(alpha)
                group['diff'].append(0)
                group['error'].append(0)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def quantize(self):
        f = []
        Q = []
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                alpha = group['alpha'][idx]
                fully_quantized = self.quantize_weight(p.data, alpha)
                group['diff'][idx] = torch.sign(p.data - fully_quantized)
                e = (p.data - fully_quantized).abs().sum()/p.data.abs().sum()
                f.append(self.e_func(e))
                Q.append(fully_quantized)
                
        if self.r_steps[self.idx] != 1:
            f = torch.FloatTensor(f).cuda()
            pr = self.prob(f)
            r_it = int(self.r_steps[self.idx]*len(f))
            index_used = []
            for _ in range(r_it):
                p_norm = pr/pr.sum()
                v = torch.rand(1).cuda()
                s, j = p_norm[0], 0
                while s < v and j + 1 < len(p):
                    j += 1
                    s += p_norm[j]
                pr[j] = 0
                index_used.append(j)
                
        count = 0
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                alpha = group['alpha'][idx]
                fully_quantized = self.quantize_weight(p.data, alpha)
                if self.r_steps[self.idx] != 1:
                    if count not in index_used:
                        group['Ts'][idx] = torch.zeros_like(p.data)
                        T = group['Ts'][idx]
                        p.data = torch.where(T == 0, fully_quantized, p.data)
                    count += 1
                else:
                    group['Ts'][idx] = torch.zeros_like(p.data)
                    T = group['Ts'][idx]
                    p.data = torch.where(T == 0, fully_quantized, p.data)


    def e_func(self, e):
        if self.e_type == "one_minus_invert":
            return 1/e + 10**(-7)
        elif self.e_type == "default":
            return e

    def prob(self, f):
        if self.prob_type == "constant":
            prob = torch.full(f.size(), 1/f.nelement())
        elif self.prob_type == "linear":
            prob = f/f.sum()
        elif self.prob_type == "softmax":
            prob = torch.exp(f)/(torch.exp(f).sum())
        elif self.prob_type == "sigmoid":
            prob = 1/(1 + torch.exp(-f))
        
        if self.e_type == "one_minus_invert":
            return 1 - prob
        elif self.e_type == "default":
            return prob

    def quantize_weight(self, tensor, alpha):
        a = (tensor > 0.5*alpha).float()
        b = (tensor < -0.5*alpha).float()
        return alpha*(a - b)

    def step(self):
        self.quantize()
        self.idx += 1


class SQ_ELQScheduler_custom_filter(object):
    def __init__(self, optimizer, r_steps, prob_type, e_type, strategy="pruning"):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if strategy not in "pruning":
            raise ValueError("ELQ supports \"pruning\" -inspired weight partitioning")
        self.optimizer = optimizer
        self.r_steps = r_steps
        self.strategy = strategy
        self.idx = 0
        self.prob_type = prob_type
        self.e_type = e_type

        for group in self.optimizer.param_groups:
            group['alpha'] = []
            group['Ts'] = []
            group['diff'] = []
            group['error'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['Ts'].append(0)
                    continue
                T = torch.ones_like(p.data).cuda()
                group['Ts'].append(T)
                alpha = torch.mean(p.data.abs()) + 0.05*torch.max(p.data.abs())
                group['alpha'].append(alpha)
                group['diff'].append(0)
                group['error'].append(0)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def quantize(self):
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                alpha = group['alpha'][idx]
                fully_quantized = self.quantize_weight(p.data, alpha)
                group['diff'][idx] = torch.sign(p.data - fully_quantized)

                if self.r_steps[self.idx] != 1:
                    f = torch.empty(p.data.size()[0])
                    for i in range(p.data.size()[0]):
                        e = (p.data[i] - fully_quantized[i]).abs().sum()/p.data[i].abs().sum()
                        f[i] = self.e_func(e)
                    prob = self.prob(f)
                    Q_ = self.Roulette(self.r_steps[self.idx], p.data, prob, fully_quantized)
                    group['Ts'][idx] = torch.where(Q_ == fully_quantized, 0, 1)
                    T = group['Ts'][idx]
                    p.data = torch.where(T == 0, fully_quantized, p.data)
                else:
                    group['Ts'][idx] = torch.zeros_like(p.data)
                    T = group['Ts'][idx]
                    p.data = torch.where(T == 0, fully_quantized, p.data)

    def e_func(self, e):
        if self.e_type == "one_minus_invert":
            return 1/e + 10**(-7)
        elif self.e_type == "default":
            return e

    def prob(self, f):
        if self.prob_type == "constant":
            prob = self.constant(f)
        elif self.prob_type == "linear":
            prob = self.linear(f)
        elif self.prob_type == "softmax":
            prob = self.softmax(f)
        elif self.prob_type == "sigmoid":
            prob = self.sigmoid(f)
        
        if self.e_type == "one_minus_invert":
            return 1 - prob
        elif self.e_type == "default":
            return prob

    def constant(self, f):
        return torch.full(f.size(), 1/f.nelement())

    def linear(self, f):
        return f/f.sum()
    
    def softmax(self, f):
        return torch.exp(f)/(torch.exp(f).sum())
    
    def sigmoid(self, f):
        return 1/(1 + torch.exp(-f))
    
    def Roulette(self, r, W, p, Q):
        Q_ = torch.empty(W.size())
        index_used = []
        r_it = int((1-r)*W.size()[0])
        for _ in range(r_it):
            p_norm = p/p.sum()
            v = torch.rand(1)
            s, j = p_norm[0], 0
            while s < v and j + 1 < len(p):
                j += 1
                s += p_norm[j]
            p[j] = 0
            Q_[j] = W[j]
            index_used.append(j)
        for index in range(W.size()[0]):
            if index not in index_used:
                Q_[index] = Q[index]
        return Q_.to("cuda")

    def quantize_weight(self, tensor, alpha):
        a = (tensor > 0.5*alpha).float()
        b = (tensor < -0.5*alpha).float()
        return alpha*(a - b)

    def step(self):
        self.quantize()
        self.idx += 1


class INQScheduler(object):
    """Handles the the weight partitioning and group-wise quantization stages
    of the incremental network quantization procedure.

    Args:
        optimizer (Optimizer): Wrapped optimizer (use inq.SGD).
        iterative_steps (list): accumulated portions of quantized weights.
        strategy ("random"|"pruning"): weight partition strategy, either random or pruning-inspired.

    Example:
        >>> optimizer = inq.SGD(...)
        >>> inq_scheduler = INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
        >>> for inq_step in range(3):
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         train(...)
        >>> inq_scheduler.step()
        >>> validate(...)

    """
    def __init__(self, optimizer, iterative_steps, strategy="pruning"):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step should equal 1 in INQ.")
        if strategy not in ["random", "pruning"]:
            raise ValueError("INQ supports \"random\" and \"pruning\" -inspired weight partitioning")
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.strategy = strategy
        self.idx = 0

        for group in self.optimizer.param_groups:
            group['ns'] = []
            if group['weight_bits'] is None:
                continue
            for p in group['params']:
                if p.requires_grad is False:
                    group['ns'].append((0, 0))
                    continue
                s = torch.max(torch.abs(p.data)).item()
                n_1 = math.floor(math.log((4*s)/3, 2))
                n_2 = int(n_1 + 1 - (2**(group['weight_bits']-1))/2)
                group['ns'].append((n_1, n_2))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def quantize(self):
        """Quantize the parameters handled by the optimizer.
        """
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                T = group['Ts'][idx]
                ns = group['ns'][idx]
                device = p.data.device
                quantizer = partial(self.quantize_weight, n_1=ns[0], n_2=ns[1])
                fully_quantized = p.data.clone().cpu().apply_(quantizer).to(device)
                p.data = torch.where(T == 0, fully_quantized, p.data)

    def quantize_weight(self, weight, n_1, n_2):
        """Quantize a single weight using the INQ quantization scheme.
        """
        alpha = 0
        beta = 2 ** n_2
        abs_weight = math.fabs(weight)
        quantized_weight = 0

        for i in range(n_2, n_1 + 1):
            if (abs_weight >= (alpha + beta) / 2) and abs_weight < (3*beta/2):
                quantized_weight = math.copysign(beta, weight)
            alpha = 2 ** i
            beta = 2 ** (i + 1)
        return quantized_weight

    def step(self):
        """Performs weight partitioning and quantization
        """
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                if group['weight_bits'] is None:
                    continue
                if self.strategy == "random":
                    if self.idx == 0:
                        probability = self.iterative_steps[0]
                    elif self.idx >= len(self.iterative_steps) - 1:
                        probability = 1
                    else:
                        probability = (self.iterative_steps[self.idx] - self.iterative_steps[self.idx - 1]) / (1 - self.iterative_steps[self.idx - 1])

                    T = group['Ts'][idx]
                    T_rand = torch.rand_like(p.data)
                    zeros = torch.zeros_like(p.data)
                    T = torch.where(T_rand <= probability, zeros, T)
                    group['Ts'][idx] = T
                else:
                    zeros = torch.zeros_like(p.data)
                    ones = torch.ones_like(p.data)
                    quantile = np.quantile(torch.abs(p.data.cpu()).numpy(), 1 - self.iterative_steps[self.idx])
                    T = torch.where(torch.abs(p.data) >= quantile, zeros, ones)
                    group['Ts'][idx] = T
                    print("Step: {}".format(self.iterative_steps[self.idx]))

        self.idx += 1
        self.quantize()


def reset_lr_scheduler(scheduler):
    scheduler.base_lrs = list(map(lambda group: group['initial_lr'], scheduler.optimizer.param_groups))
    last_epoch = 0
    scheduler.last_epoch = last_epoch
    scheduler.step(last_epoch)