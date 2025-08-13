import torch
import torch.nn.functional as F


class BatchNormalizedMLP:
    """
    This is a 3-layer MLP with the following required arguments:
        h: Total number of neurons in the hidden layer
        n: Number of characters provided as context
    """
    def __init__(self, h:int, n:int, feature_dims:int)->None:
        self.h = h
        self.n = n
        self.feature_dims = feature_dims
        self.vocab_size = 27 #26 alphabets + 1 special token(".")
        self.generator = torch.Generator().manual_seed(6385189022)

        self.C = torch.randn((self.vocab_size, feature_dims), generator=self.generator, requires_grad= True)
        self.H = torch.randn((n*feature_dims,h), generator=self.generator, requires_grad=True)
        self.b1 = torch.randn(h, generator=self.generator, requires_grad=True)
        self.W1 = torch.randn((h,self.vocab_size), generator=self.generator, requires_grad=True)
        self.W2 = torch.randn((n*feature_dims,self.vocab_size), generator=self.generator, requires_grad=True)
        self.b2 = torch.randn(self.vocab_size, generator=self.generator, requires_grad=True)

        self.cross_entropy_loss = torch.tensor(0)


    def _kaiming_init_all_weights(self)->None:
        """
        IF NEEDED
        Since I'm using tanh() act func across all hidden layer neurons my kaiming init factor is: (5/3)*1/sqrt(fan_in)
        """
        self.H *= (5/3)*(1/self.n*self.feature_dims)
    
    def _squash_op_layer_params(self)->None:
        """
        Minimize the output layer parameters by a factor b/w [0,1]
        to ensure that the logits produced are close to zero(at initialiation our network makes no assumptions)
        """
        self.W1 *= 0.01
        self.W2 *= 0.01
        self.b2 *= 0
    
    def forward(self, x_train:torch.Tensor)->torch.Tensor:
        emb = self.C[x_train]
        emb = emb.view(-1,self.n*self.feature_dims)
        hpreact = emb@self.H
        h = (hpreact+self.b1).tanh()

        l1 = h@self.W1
        l2 = emb@self.W2 + self.b2
        logits = l1+l2
        return logits

    def gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        #optional weights initializations done
        self._kaiming_init_all_weights()
        self._squash_op_layer_params()
        
        params = [self.C, self.H, self.b1, self.W1, self.W2, self.b2]

        for _ in range(epochs):
            #forward pass
            logits = self.forward(x_train)

            #loss computation
            self.cross_entropy_loss = F.cross_entropy(logits, y_train, reduction='mean', label_smoothing=reg_factor)

            #backward pass
            self.cross_entropy_loss.backward()

            #grad update
            for param in params:
                param -= h*param.grad
                param.grad = None
    
    def stochastic_gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        pass

    def minibatch_gradient_descent(self, minibatch_size:int, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        pass