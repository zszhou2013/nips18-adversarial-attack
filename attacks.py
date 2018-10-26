import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf

# --- White-box attacks ---


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.step_size = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        
        x = X_nat.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            
        for i in range(self.k):
            
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, X_nat - self.epsilon), X_nat + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return x
    
    
class TransferAttack(object):
    
    def __init__(self, ori_model=None, models=None, epsilon=16/255, k=10, a=0.01, 
        random_start=True):
        
        self.ori_model = ori_model
        self.models = models
        
        self.epsilon = epsilon
        self.k = k
        self.random_start = random_start
        self.step_size = a
        
        self.loss_fn = nn.CrossEntropyLoss()
        
      
    def attack_batch(self, batch_tensor, y, target_attack=True):
        """inputs are pytorch tensor format
        if target is None, we use the untarget attack method.
        """
        
        x = batch_tensor.detach()
        
            
        if self.random_start:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
                
        for i in range(self.k):             
            #get grad            
            x.requires_grad_()
            if self.models is None:
                with torch.enable_grad():
                    logits = self.ori_model(x)
                    loss = self.loss_fn(logits, y)
                grad = torch.autograd.grad(loss, [x])[0]
                final_grad = grad.detach()              
            else:
                grads = []
                for j in range(len(self.models)):
                    with torch.enable_grad():
                        logits = self.models[j](x)
                        loss = self.loss_fn(logits, y)
                    grad = torch.autograd.grad(loss, [x])[0].detach()
                    grads.append(grad)
                final_grad = torch.zeros_like(batch_tensor)  
                for g in grads:
                    final_grad += g
                final_grad /= len(self.models)        
                
            if not target_attack:    
                x = x.detach() + self.step_size*torch.sign(final_grad)
            else:
                x = x.detach() - self.step_size*torch.sign(final_grad)
           
            x = torch.min(torch.max(x, batch_tensor - self.epsilon), batch_tensor + self.epsilon)
            x = torch.clamp(x, 0, 1)                    

        return x
    
    
    def attack_tensor(self, tensor, y, target_attack=True):
        
        batch_tensor = tensor.unsqueeze(0)
        
        #y = y.unsqueeze(0)
            
        x = self.attack_batch(batch_tensor, y,target_attack=target_attack)
        x = x.squeeze(0)
        
        return x
        
        
    def attack_cv2(self, cv2_image, y,device, target_attack=True):
        
        tensor = ttf.to_tensor(cv2_image)
        y = torch.LongTensor([y])        
        
        tensor = tensor.to(device)
        y = y.to(device)
        
        x = self.attack_tensor(tensor, y,target_attack=target_attack)
        
        pic = x.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))
        
        return npimg    