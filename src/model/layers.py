import torch
from enum import StrEnum
from typing import Iterator, Literal, Self, Optional


class RUN_TYPE(StrEnum):
    TRAIN = "train"
    INFERENCE = "inference"
class OPTIMIZATION_TYPE(StrEnum):
    GD = "gradient_descent"
    SGD = "stochastic_gradient_descent"
    MINIBATCH_GD = "minibatch_gradient_descent"

class Module:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(6385189022)
    def __call__(self):
        pass

class Linear(Module):
    def __init__(self, fan_in:int, fan_out:int, bias:bool = True)->None:
        super().__init__()
        self.W = torch.randn((fan_in,fan_out), generator=self.generator)
        self.b = torch.zeros(fan_out) if bias else None
    
    def with_kaiming_init(self, nonlinearity:Literal['tanh', 'relu', 'leaky_relu'], neg_slope:Optional[float])->Self:
        gain = {
            "tanh":5/3,
            "relu":2**0.5,
            "leaky_relu": (2/(1+neg_slope**2))**0.5
        }.get(nonlinearity, 1)
        self.W.data *= (gain/len(self.W)**0.5)
        return self
    
    def scale_down_weights(self, factor:Optional[float]=0.01)->Self:
        """
        Minimize the output layer parameters by a factor b/w [0,1]
        to ensure that the logits produced are close to zero(at initialiation our network makes no assumptions)
        """
        self.W.data *= factor
        return self
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        self.out = x@self.W 
        if self.b is not None:
            self.out += self.b
        return self.out

    @property
    def params(self)->Iterator[torch.Tensor]:
        params = [self.W]
        if self.b is not None:
            return params+[self.b]
        return iter(params)

class Tanh(Module):
    def __call__(self,x:torch.Tensor)->torch.Tensor:
        self.out = torch.tanh(x)
        return self.out
    
    @property
    def params(self)->Iterator[torch.Tensor]:
        return iter([])

class BatchNorm1d(Module):
    """
    LESSON LEARNT: With a NoneType return you will definitely lose the child nodes of hpreact
    that are added within this function. if you want to retain child nodes, return hpreact.
    """
    def __init__(self, dim:int, momentum:float=0.001, epsilon:float = 1e-5)->None:
        self.optim_type = OPTIMIZATION_TYPE.MINIBATCH_GD
        self.run_type = RUN_TYPE.TRAIN

        self.epsilon = epsilon
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.momentum = momentum
        self.running_mean = torch.zeros(dim)
        self.running_std = torch.ones(dim)

    def __call__(self, x:torch.Tensor)->torch.Tensor:
        if self.optim_type!=OPTIMIZATION_TYPE.MINIBATCH_GD: return

        if self.run_type==RUN_TYPE.TRAIN:
            bn_mean, bn_std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + self.epsilon
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*bn_mean
                self.running_std = (1-self.momentum)*self.running_std + self.momentum*bn_std
        else:
            bn_mean, bn_std = self.running_mean, self.running_std

        norm_preact = (x-bn_mean)/bn_std
        self.out = norm_preact*self.gamma + self.beta
        return self.out

    @property
    def params(self)->Iterator[torch.Tensor]:
        return iter([self.gamma, self.beta])


class NgramEmbeddingTable(Module):
    def __init__(self, n:int, vocab_size:int, feature_dims:int)->None:
        super().__init__()
        self.n = n
        self.feature_dims = feature_dims
        self.C = torch.randn((vocab_size, feature_dims), generator=self.generator)
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        emb = self.C[x]
        return emb.view(-1, self.n*self.feature_dims)

    @property
    def params(self)->Iterator[torch.Tensor]:
        return iter([self.C])