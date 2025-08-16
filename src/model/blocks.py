import torch
from typing import Iterator

from layers import Module

class Sequential(Module):
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
    
    @property
    def params(self)->Iterator[torch.Tensor]:
        params = []
        for layer in self._modules:
            params.extend(layer.params)
        return iter(params)
    
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