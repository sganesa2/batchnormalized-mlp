from typing import Iterator

import torch
import torch.nn.functional as F

from layers import (
    NgramEmbeddingTable, Linear, BatchNorm1d, Tanh, RUN_TYPE, OPTIMIZATION_TYPE
)
from blocks import Sequential

class BatchNormalizedMLP:
    """
    This is a 3-layer MLP with the following required arguments:
        h: Total number of neurons in the hidden layer
        n: Number of characters provided as context
    """
    def __init__(self, h:int, n:int, feature_dims:int)->None:
        self.vocab_size = 27 #26 alphabets + 1 special token(".")

        self.sequential_layer1 = Sequential(
            NgramEmbeddingTable(n, self.vocab_size, feature_dims),
            Linear(n*feature_dims, h, bias = False),
            BatchNorm1d(h),
            Tanh(),
            Linear(h, self.vocab_size)
        )
        self.sequential_layer2 = Sequential(
            NgramEmbeddingTable(n, self.vocab_size, feature_dims),
            Linear(n*feature_dims, self.vocab_size)
        )

        self.cross_entropy_loss = torch.tensor(0)
    
    @property
    def params(self)->Iterator[torch.Tensor]:
        pass

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out1, out2 = self.sequential_layer1(x), self.sequential_layer2(x)
        logits = out1+out2
        return logits

    def eval(self)->None:
        for layer in self.sequential_layer1:
            if isinstance(layer, BatchNorm1d):
                layer.run_type = RUN_TYPE.INFERENCE
                layer.optim_type = None

    def _training_code(self, x:torch.Tensor, y:torch.Tensor, h:float, reg_factor:float)->None:
        #zero grad
        for param in self.params:
            param.grad = None

        #forward pass
        logits = self.forward(x)

        #loss computation
        self.cross_entropy_loss = F.cross_entropy(logits, y, reduction='mean', label_smoothing=reg_factor)

        #backward pass
        self.cross_entropy_loss.backward()

        #grad update
        for param in self.params:
            param.data -= h*param.grad

    def gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        run_type, optim_type = RUN_TYPE(run_type), OPTIMIZATION_TYPE(optim_type)

        for layer in self.sequential_layer1:
            if isinstance(layer, BatchNorm1d):
                layer.run_type = run_type
                layer.optim_type = optim_type

        for _ in range(epochs):
            self._training_code(x_train,y_train,h,reg_factor, "gradient_descent")
    
    def stochastic_gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        # self._kaiming_init_all_weights()
        self._squash_op_layer_params()

        for _ in range(epochs):
            for example,label in zip(x_train, y_train):
                self._training_code(example,label,h,reg_factor, "stochastic_gradient_descent")


    def minibatch_gradient_descent(self, minibatch_size:int, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        self._kaiming_init_all_weights()
        self._squash_op_layer_params()

        permutes = torch.randperm(x_train.shape[0], generator=self.generator)
        x_train, y_train = x_train[permutes], y_train[permutes]
        x_train_minibatches, y_train_minibatches = x_train.split(minibatch_size), y_train.split(minibatch_size)

        for _ in range(epochs):
            for x,y in zip(x_train_minibatches,y_train_minibatches):
                self._training_code(x,y,h,reg_factor, "minibatch_gradient_descent")