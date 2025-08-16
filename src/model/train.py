import torch
import torch.nn.functional as F

from model.layers import (
    NgramEmbeddingTable, Linear, BatchNorm1d, Tanh, RUN_TYPE
)
from model.blocks import Sequential

class BatchNormalizedMLP:
    """
    This is a 3-layer MLP with the following required arguments:
        h: Total number of neurons in the hidden layer
        n: Number of characters provided as context
    """
    def __init__(self, h:int, n:int, feature_dims:int)->None:
        self.n =n
        self.vocab_size = 27 #26 alphabets + 1 special token(".")
        self.generator = torch.Generator().manual_seed(6385189022)
        self.embedding_table = NgramEmbeddingTable(n, self.vocab_size, feature_dims)
        self.sequential_layer1 = Sequential(
            self.embedding_table,
            Linear(n*feature_dims, h, bias = False),
            BatchNorm1d(h),
            Tanh(),
            Linear(h, self.vocab_size)
        )
        self.sequential_layer2 = Sequential(
            self.embedding_table,
            Linear(n*feature_dims, self.vocab_size)
        )
        self.params = [*self.sequential_layer1.params, *self.sequential_layer2.params]
        self.cross_entropy_loss = torch.tensor(0)

        for p in self.params:
            p.requires_grad = True

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out1, out2 = self.sequential_layer1(x), self.sequential_layer2(x)
        logits = out1+out2
        return logits

    def eval(self)->None:
        for layer in self.sequential_layer1:
            if isinstance(layer, BatchNorm1d):
                layer.run_type = RUN_TYPE.INFERENCE

    def _remove_batchnorm(self)->None:
        for layer in self.sequential_layer1:
            if isinstance(layer, BatchNorm1d):
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
        self._remove_batchnorm()
        for _ in range(epochs):
            self._training_code(x_train,y_train,h,reg_factor)
    
    def stochastic_gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        self._remove_batchnorm()
        for _ in range(epochs):
            for example,label in zip(x_train, y_train):
                self._training_code(example,label,h,reg_factor)

    def minibatch_gradient_descent(self, minibatch_size:int, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        permutes = torch.randperm(x_train.shape[0], generator=self.generator)
        x_train, y_train = x_train[permutes], y_train[permutes]
        x_train_minibatches, y_train_minibatches = x_train.split(minibatch_size), y_train.split(minibatch_size)

        for _ in range(epochs):
            for x,y in zip(x_train_minibatches,y_train_minibatches):
                self._training_code(x,y,h,reg_factor)