import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_model(model, acc, name):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict()
    }
    torch.save(state, f"trained_models/{name}")
    print('*** DONE! ***')


class FWN:
    def __init__(self, model):
        pass

    def Quantization(self, r):
        pass

    def Restore(self):
        pass


class BWN:
    def __init__(self,model):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.binarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.binarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def BinarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.Binarize(self.target_modules[index].data)
    
    def Binarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)

    def Alpha(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     #convolution layer
            alpha = tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   #fc layer
            alpha = tensor.norm(1,1).div(n)
        return alpha
            
    def Quantization(self, r):
        self.SaveWeights()
        self.BinarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_BWN_default_layer:
    def __init__(self, model, prob_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.binarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.binarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type

    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def SQ_BinarizeWeights(self, r):
        f = []
        Q = []
        for index in range(self.num_of_params):
            f_temp, Q_temp = self.Binarize(self.target_modules[index].data)
            f.append(f_temp)
            Q.append(Q_temp)
        f = torch.FloatTensor(f).cuda()
        p = self.prob(f)
        r_it = int(r*self.num_of_params)
        index_used = []
        for _ in range(r_it):
            p_norm = p/p.sum()
            v = torch.rand(1).cuda()
            s, j = p_norm[0], 0
            while s < v and j + 1 < len(p):
                j += 1
                s += p_norm[j]
            p[j] = 0
            index_used.append(j)
            self.target_modules[j].data = Q[j].data

    def prob(self, f):
        if self.prob_type == "constant":
            prob = torch.full(f.size(), 1/f.nelement())
        elif self.prob_type == "linear":
            prob = f/f.sum()
        elif self.prob_type == "softmax":
            prob = torch.exp(f)/(torch.exp(f).sum())
        elif self.prob_type == "sigmoid":
            prob = 1/(1 + torch.exp(-f))
        return prob

    def BinarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Binarize(self.target_modules[index].data)
    
    def No_SQ_Binarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)
    
    def Binarize(self, tensor):
        tensor = tensor.to("cpu")
        Q = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            Q[i] = alpha[i]*(pos_one - neg_one)
        e = (tensor - Q).abs().sum()/tensor.abs().sum()
        f = 1/e + 10**(-7)
        return f, Q.to(device)
        
    def Alpha(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     #convolution layer
            alpha = tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   #fc layer
            alpha = tensor.norm(1,1).div(n)
        return alpha
    
    def Quantization(self, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_BinarizeWeights(r)
        else:
            self.BinarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_BWN_custom_layer:
    def __init__(self, model, prob_type, e_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.binarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.binarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type
        self.e_type = e_type

    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def SQ_BinarizeWeights(self, r):
        f = []
        Q = []
        for index in range(self.num_of_params):
            f_temp, Q_temp = self.Binarize(self.target_modules[index].data)
            f.append(f_temp)
            Q.append(Q_temp)
        f = torch.FloatTensor(f).cuda()
        p = self.prob(f)
        r_it = int((1-r)*self.num_of_params)
        index_used = []
        for _ in range(r_it):
            p_norm = p/p.sum()
            v = torch.rand(1).cuda()
            s, j = p_norm[0], 0
            while s < v and j + 1 < len(p):
                j += 1
                s += p_norm[j]
            p[j] = 0
            index_used.append(j)
        for index in range(self.num_of_params):
            if index not in index_used:
                self.target_modules[index].data = Q[index].data

    def BinarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Binarize(self.target_modules[index].data)
    
    def No_SQ_Binarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)
    
    def Binarize(self, tensor):
        tensor = tensor.to("cpu")
        Q = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            Q[i] = alpha[i]*(pos_one - neg_one)
        e = (tensor - Q).abs().sum()/tensor.abs().sum()
        if self.e_type == "one_minus_invert":
            f = 1/e + 10**(-7)
        else:
            f = e
        return f, Q.to(device)

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
        
    def Alpha(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     #convolution layer
            alpha = tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   #fc layer
            alpha = tensor.norm(1,1).div(n)
        return alpha
    
    def Quantization(self, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_BinarizeWeights(r)
        else:
            self.BinarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_BWN_custom_filter:
    def __init__(self, model, prob_type, e_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.binarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.binarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type
        self.e_type = e_type
    
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def SQ_BinarizeWeights(self, r):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.SQ_Binarize(self.target_modules[index].data, r)

    def BinarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Binarize(self.target_modules[index].data)
    
    def No_SQ_Binarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)
    
    def Binarize(self, tensor):
        f = torch.empty(tensor.size()[0])
        Q = torch.empty(tensor.size())
        alpha = self.Alpha(tensor)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > 0).float()
            neg_one = (w < -0).float()
            Q[i] = alpha[i]*(pos_one - neg_one)
            e = (w - Q[i]).abs().sum()/w.abs().sum()
            if self.e_type == "one_minus_invert":
                f[i] = 1/e + 10**(-7)
            else:
                f[i] = e
        return f, Q

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
        return Q_
    
    def SQ_Binarize(self, W, r):
        W = W.to("cpu")
        f, Q = self.Binarize(W)
        p = self.prob(f)
        Q_ = self.Roulette(r, W, p, Q)
        return Q_.to(device)

    def Alpha(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     #convolution layer
            alpha = tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   #fc layer
            alpha = tensor.norm(1,1).div(n)
        return alpha
    
    def Quantization(self, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_BinarizeWeights(r)
        else:
            self.BinarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class TWN:
    def __init__(self,model):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
       
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.Ternarize(self.target_modules[index].data)
    
    def Ternarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.zeros(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta[i]).float()
            neg_one = (tensor[i] < -delta[i]).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)

    def Alpha(self,tensor,delta):
        alpha = torch.empty(tensor.size()[0], 1)
        for i in range(tensor.size()[0]):
            absvalue = tensor[i].abs()
            truth_value = (absvalue > delta[i]).float()
            count = truth_value.sum()
            abssum = (absvalue*truth_value).sum()
            alpha[i] = abssum/count
        return alpha

    def Delta(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     #convolution layer
            delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   #fc layer
            delta = 0.7 * tensor.norm(1,1).div(n)
        return delta
    
    def Quantization(self, r):
        self.SaveWeights()
        self.TernarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_TWN_default_layer:
    def __init__(self, model, prob_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type

    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def SQ_TernarizeWeights(self, r):
        f = []
        Q = []
        for index in range(self.num_of_params):
            f_temp, Q_temp = self.Ternarize(self.target_modules[index].data)
            f.append(f_temp)
            Q.append(Q_temp)
        f = torch.FloatTensor(f).cuda()
        p = self.prob(f)
        r_it = int(r*self.num_of_params)
        index_used = []
        for _ in range(r_it):
            p_norm = p/p.sum()
            v = torch.rand(1).cuda()
            s, j = p_norm[0], 0
            while s < v and j + 1 < len(p):
                j += 1
                s += p_norm[j]
            p[j] = 0
            index_used.append(j)
            self.target_modules[j].data = Q[j].data

    def prob(self, f):
        if self.prob_type == "constant":
            prob = torch.full(f.size(), 1/f.nelement())
        elif self.prob_type == "linear":
            prob = f/f.sum()
        elif self.prob_type == "softmax":
            prob = torch.exp(f)/(torch.exp(f).sum())
        elif self.prob_type == "sigmoid":
            prob = 1/(1 + torch.exp(-f))
        return prob

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Ternarize(self.target_modules[index].data)
    
    def Ternarize(self, tensor):
        tensor = tensor.to("cpu")
        f = torch.empty(tensor.size()[0])
        Q = torch.empty(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > delta[i]).float()
            neg_one = (w < -delta[i]).float()
            Q[i] = alpha[i]*(pos_one - neg_one)
        e = (tensor - Q).abs().sum()/w.abs().sum()
        f = 1/e + 10**(-7)
        return f, Q.to(device)
    
    def No_SQ_Ternarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta[i]).float()
            neg_one = (tensor[i] < -delta[i]).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)

    def Alpha(self,tensor,delta):
        alpha = torch.empty(tensor.size()[0], 1)
        for i in range(tensor.size()[0]):
            absvalue = tensor[i].abs()
            truth_value = (absvalue > delta[i]).float()
            count = truth_value.sum()
            abssum = (absvalue*truth_value).sum()
            alpha[i] = abssum/count
        return alpha

    def Delta(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     # convolution layer
            delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   # fc layer
            delta = 0.7 * tensor.norm(1,1).div(n)
        return delta
    
    def Quantization(self, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_TernarizeWeights(r)
        else:
            self.TernarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_TWN_custom_layer:
    def __init__(self, model, prob_type, e_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type
        self.e_type = e_type

    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def SQ_TernarizeWeights(self, r):
        f = []
        Q = []
        for index in range(self.num_of_params):
            f_temp, Q_temp = self.Ternarize(self.target_modules[index].data)
            f.append(f_temp)
            Q.append(Q_temp)
        f = torch.FloatTensor(f).cuda()
        p = self.prob(f)
        r_it = int(r*self.num_of_params)
        index_used = []
        for _ in range(r_it):
            p_norm = p/p.sum()
            v = torch.rand(1).cuda()
            s, j = p_norm[0], 0
            while s < v and j + 1 < len(p):
                j += 1
                s += p_norm[j]
            p[j] = 0
            index_used.append(j)
        for index in range(self.num_of_params):
            if index not in index_used:
                self.target_modules[index].data = Q[index].data

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

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Ternarize(self.target_modules[index].data)
    
    def Ternarize(self, tensor):
        tensor = tensor.to("cpu")
        f = torch.empty(tensor.size()[0])
        Q = torch.empty(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > delta[i]).float()
            neg_one = (w < -delta[i]).float()
            Q[i] = alpha[i]*(pos_one - neg_one)
        e = (w - Q).abs().sum()/w.abs().sum()
        if self.e_type == "one_minus_invert":
            f[i] = 1/e + 10**(-7)
        else:
            f[i] = e
        return f, Q.to(device)
    
    def No_SQ_Ternarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta[i]).float()
            neg_one = (tensor[i] < -delta[i]).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)

    def Alpha(self,tensor,delta):
        alpha = torch.empty(tensor.size()[0], 1)
        for i in range(tensor.size()[0]):
            absvalue = tensor[i].abs()
            truth_value = (absvalue > delta[i]).float()
            count = truth_value.sum()
            abssum = (absvalue*truth_value).sum()
            alpha[i] = abssum/count
        return alpha

    def Delta(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     # convolution layer
            delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   # fc layer
            delta = 0.7 * tensor.norm(1,1).div(n)
        return delta
    
    def Quantization(self, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_TernarizeWeights(r)
        else:
            self.TernarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_TWN_custom_filter:
    def __init__(self, model, prob_type, e_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type
        self.e_type = e_type
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def SQ_TernarizeWeights(self, r):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.SQ_Ternarize(self.target_modules[index].data, r)

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Ternarize(self.target_modules[index].data)
    
    def No_SQ_Ternarize(self,tensor):
        tensor = tensor.to("cpu")
        output = torch.empty(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta[i]).float()
            neg_one = (tensor[i] < -delta[i]).float()
            output[i] = alpha[i]*(pos_one - neg_one)
        return output.to(device)
    
    def Ternarize(self, tensor):
        tensor = tensor.to("cpu")
        f = torch.empty(tensor.size()[0])
        Q = torch.empty(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            w = tensor[i]
            pos_one = (w > delta[i]).float()
            neg_one = (w < -delta[i]).float()
            Q[i] = alpha[i]*(pos_one - neg_one)
            e = (w - Q[i]).abs().sum()/w.abs().sum()
            if self.e_type == "one_minus_invert":
                f[i] = 1/e + 10**(-7)
            else:
                f[i] = e
        return f, Q

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
        return Q_
    
    def SQ_Ternarize(self, W, r):
        W = W.to("cpu")
        f, Q = self.Ternarize(W)
        p = self.prob(f)
        Q_ = self.Roulette(r, W, p, Q)
        return Q_.to(device)

    def Alpha(self,tensor,delta):
        alpha = torch.empty(tensor.size()[0], 1)
        for i in range(tensor.size()[0]):
            absvalue = tensor[i].abs()
            truth_value = (absvalue > delta[i]).float()
            count = truth_value.sum()
            abssum = (absvalue*truth_value).sum()
            alpha[i] = abssum/count
        return alpha

    def Delta(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     # convolution layer
            delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   # fc layer
            delta = 0.7 * tensor.norm(1,1).div(n)
        return delta
            
    def Quantization(self, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_TernarizeWeights(r)
        else:
            self.TernarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class Trained_TernarizeOp:
    def __init__(self,model):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter

    def Ternarize(self, tensor, sf):
        t = 0.15
        delta = t*tensor.abs().max()
        a = (tensor > delta).float()
        b = (tensor < -delta).float()
        return sf[0]*a + (-sf[1]*b)
    
    def UpdateGradients(self, quant_tensor, tensor, sf):
        weight_grad, w_p_grad, w_n_grad = self.GetGrads(quant_tensor.grad, tensor.data, 
                                                        sf.data[0], sf.data[1], 0.15)
        
        tensor.grad = Variable(weight_grad)
        sf.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).cuda())
        quant_tensor.grad.data.zero_()
        return tensor, sf

    def GetGrads(self, quant_tensor_grad, tensor, w_p, w_n, t):
        delta = t*tensor.abs().max()
        # masks
        a = (tensor > delta).float()
        b = (tensor < -delta).float()
        c = torch.ones(tensor.size()).cuda() - a - b
        # scaled tensor grad and grads for scaling factors (w_p, w_n)
        return w_p*a*quant_tensor_grad + w_n*b*quant_tensor_grad + 1.0*c*quant_tensor_grad,\
            (a*quant_tensor_grad).sum(), (b*quant_tensor_grad).sum()
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def TernarizeWeights(self, scaling_factors):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.Ternarize(self.target_modules[index], scaling_factors[index])

    def Quantization(self, scaling_factors, r):
        self.SaveWeights()
        self.TernarizeWeights(scaling_factors)
    
    def UpdateGradientsAndRestore(self, scaling_factors):
        for index in range(self.num_of_params):
            self.target_modules[index], scaling_factors[index] = self.UpdateGradients(self.target_modules[index],
                                                                                      self.saved_params[index].data,
                                                                                      scaling_factors[index])
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


class SQ_Trained_TernarizeOp:
    def __init__(self,model, prob_type, e_type):
        model = model.to(device)
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
        self.prob_type = prob_type
        self.e_type = e_type

    def SQ_TernarizeWeights(self, scaling_factors, r):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.SQ_Ternarize(self.target_modules[index], scaling_factors[index], r)

    def TernarizeWeights(self, scaling_factors):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.No_SQ_Ternarize(self.target_modules[index], scaling_factors[index])

    def No_SQ_Ternarize(self, tensor, sf):
        t = 0.15
        delta = t*tensor.abs().max()
        a = (tensor > delta).float()
        b = (tensor < -delta).float()
        return sf[0]*a + (-sf[1]*b)

    def Ternarize(self, tensor, sf):
        t = 0.15
        delta = t*tensor.abs().max()
        a = (tensor > delta).float()
        b = (tensor < -delta).float()
        return sf[0]*a + (-sf[1]*b)
    
    def UpdateGradients(self, quant_tensor, tensor, sf):
        weight_grad, w_p_grad, w_n_grad = self.GetGrads(quant_tensor.grad, tensor.data, 
                                                        sf.data[0], sf.data[1], 0.15)
        
        tensor.grad = Variable(weight_grad)
        sf.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).cuda())
        quant_tensor.grad.data.zero_()
        return tensor, sf
    
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
        return Q_
    
    def SQ_Ternarize(self, W, scaling_factor, r):
        s = W.size()[0]
        Q = self.Ternarize(W, scaling_factor)
        p = torch.full((s,), 1/s)
        Q_ = self.Roulette(r, W, p, Q)
        return Q_.to(device)

    def GetGrads(self, quant_tensor_grad, tensor, w_p, w_n, t):
        delta = t*tensor.abs().max()
        # masks
        a = (tensor > delta).float()
        b = (tensor < -delta).float()
        c = torch.ones(tensor.size()).cuda() - a - b
        # scaled tensor grad and grads for scaling factors (w_p, w_n)
        return w_p*a*quant_tensor_grad + w_n*b*quant_tensor_grad + 1.0*c*quant_tensor_grad,\
            (a*quant_tensor_grad).sum(), (b*quant_tensor_grad).sum()
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def Quantization(self, scaling_factors, r):
        self.SaveWeights()
        if r != 1:
            self.SQ_TernarizeWeights(scaling_factors, r)
        else:
            self.TernarizeWeights(scaling_factors)
    
    def UpdateGradientsAndRestore(self, scaling_factors):
        for index in range(self.num_of_params):
            self.target_modules[index], scaling_factors[index] = self.UpdateGradients(self.target_modules[index],
                                                                                      self.saved_params[index].data,
                                                                                      scaling_factors[index])
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
