from typing import Literal

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
        # self.b1 = torch.randn(h, generator=self.generator, requires_grad=True)
        self.W1 = torch.randn((h,self.vocab_size), generator=self.generator, requires_grad=True)
        self.W2 = torch.randn((n*feature_dims,self.vocab_size), generator=self.generator, requires_grad=True)
        self.b2 = torch.randn(self.vocab_size, generator=self.generator, requires_grad=True)

        self.bn_epsilon = 1e-5
        self.H_bngain = torch.ones(h, requires_grad=True)
        self.H_bnbias = torch.zeros(h, requires_grad= True)
        self.H_bnmean_running = torch.zeros(h)
        self.H_bnstd_running = torch.ones(h)
        self.bnmagic_no:float = 0.999

        self.cross_entropy_loss = torch.tensor(0)

    @property
    def params(self)->list[torch.Tensor]:
        return [self.C, self.H, self.W1, self.W2, self.b2, self.H_bngain, self.H_bnbias]

    def _kaiming_init_all_weights(self)->None:
        """
        IF NEEDED
        Since I'm using tanh() act func across all hidden layer neurons my kaiming init factor is: (5/3)*1/sqrt(fan_in)
        """
        self.H.data *= (5/3)*(1/self.n*self.feature_dims)
    
    def _squash_op_layer_params(self)->None:
        """
        Minimize the output layer parameters by a factor b/w [0,1]
        to ensure that the logits produced are close to zero(at initialiation our network makes no assumptions)
        """
        self.W1.data *= 0.01
        self.W2.data *= 0.01
        self.b2.data *= 0

    def _batchnormalized(self, hpreact:torch.Tensor, flag:bool, run_type:Literal['train','inference'])->None:
        """
        LESSON LEARNT: With a NoneType return you will definitely lose the child nodes of hpreact
        that are added within this function. if you want to retain child nodes, return hpreact.
        """
        
        if not flag: return

        if run_type=='train':
            bnmean, bnstd = hpreact.mean(0, keepdim=True), hpreact.std(0, keepdim= True)+self.bn_epsilon
            with torch.no_grad():
                self.H_bnmean_running = self.bnmagic_no*self.H_bnmean_running + (1-self.bnmagic_no)*bnmean
                self.H_bnstd_running = self.bnmagic_no*self.H_bnstd_running + (1-self.bnmagic_no)*bnstd
        else:
            bnmean, bnstd = self.H_bnmean_running, self.H_bnstd_running

        hpreact = (hpreact-bnmean)/bnstd
        hpreact = (hpreact*self.H_bngain) + self.H_bnbias
        return hpreact

    
    def forward(self, xs:torch.Tensor, optim_type:str, run_type:Literal['train','inference'])->torch.Tensor:
        flag = True if optim_type=='minibatch_gradient_descent' else False
            
        emb = self.C[xs]
        emb = emb.view(-1,self.n*self.feature_dims)
        hpreact = emb@self.H
        hpreact = self._batchnormalized(hpreact, flag, run_type)
        h = hpreact.tanh()

        l1 = h@self.W1
        l2 = emb@self.W2 + self.b2
        logits = l1+l2
        return logits

    def _training_code(self, x:torch.Tensor, y:torch.Tensor, h:float, reg_factor:float, optim_type:str)->None:
        #zero grad
        for param in self.params:
            param.grad = None

        #forward pass
        logits = self.forward(x, optim_type, 'train')

        #loss computation
        self.cross_entropy_loss = F.cross_entropy(logits, y, reduction='mean', label_smoothing=reg_factor)

        #backward pass
        self.cross_entropy_loss.backward()

        #grad update
        for param in self.params:
            param.data -= h*param.grad

    def gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        #optional weights initializations done
        # self._kaiming_init_all_weights()
        self._squash_op_layer_params()

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

def stoi()->dict[str,int]:
    start_index, total_chars = 97, 26
    stoi_dict = {chr(i):i-start_index+1 for i in range(start_index, start_index+total_chars+1)}
    return {".":0, **stoi_dict}

def itos()->dict[int,str]:
    stoi_dict = stoi()
    return {v:k for k,v in stoi_dict.items()}