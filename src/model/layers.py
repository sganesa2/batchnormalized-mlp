import torch
from typing import Iterator, Literal

class Module:
    def __call__(self)->None:
        return

class Linear(Module):
    def __init__(self, fan_in:int, fan_out:int, bias:bool)->None:
        self.generator = torch.Generator().manual_seed(6385189022)
        self.W = torch.randn((fan_in,fan_out), generator=self.generator)
        self.b = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        out = x@self.W 
        if self.b is not None:
            return out + self.b
        return out

    @property
    def params(self)->list[torch.Tensor]:
        params = [self.W]
        if self.b is not None:
            return params+[self.b]
        return params

class Tanh(Module):
    def __call__(self,x:torch.Tensor)->torch.Tensor:
        self.out = torch.tanh(x)
        return self.out
    
    @property
    def params(self)->list[torch.Tensor]:
        return [self.out]

class BatchNorm1d(Module):
    def __init__(self, dim:int, run_type:Literal['train','inference'], flag:bool, momentum:float=0.001, epsilon:float = 1e-5)->None:
        self.epsilon = epsilon
        self.momentum = momentum
    
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_std = torch.ones(dim)
        #TODO: Complete rest of batchnorm implementation
        

    def __call__(self):
        pass

    @property
    def params(self)->list[torch.Tensor]:
        pass


class Sequential:
    def __init__(self, *args)->None:
        self._modules: dict[str, Module] = {}

        for idx, module in enumerate(args):
            if isinstance(module, Module):
                self._modules[idx] = module

    def __iter__(self)->Iterator[Module]:
        return iter(self._modules.values())
    
    def __len__(self)->int:
        return len(self._modules)
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x
    
    def append(self, module:Module)->None:
        if not isinstance(module, Module):
            raise TypeError("You have passed a module that isn't of type Module")
        
        self._modules[len(self)] = module
    
    def insert(self, idx:int, module:Module)->None:
        if not isinstance(module, Module):
            raise TypeError("You have passed a module that isn't of type Module")
        
        n = len(self._modules)
        if not (-n <= idx <= n):
            raise IndexError(f"Index out of range: {idx}")
        if idx<0:
            idx +=n

        for i in range(n, idx, -1):
            self._modules[i] = self._modules[i-1]
        self._modules[idx] = module

        
