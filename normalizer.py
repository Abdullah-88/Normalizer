import torch
from torch import nn


       
 

class GatingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.proj_1 =  nn.Linear(dim,dim)
        self.proj_2 =  nn.Linear(dim,dim)
        self.proj_3 = nn.Linear(dim,dim)     
        self.silu = nn.SiLU()
        
             	   
    def forward(self, x):
        u, v = x, x 
        u = self.proj_1(u)
        u = self.silu(u)
       
        
        v = self.proj_2(v)
     
       
        g = u * v
        g = self.proj_3(g)
       
        out = g
        return out



class NormalizerBlock(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()
       
         
        self.norm_global = nn.LayerNorm(d_model * num_tokens)
        self.norm_local = nn.LayerNorm(d_model)                    
        self.gating = GatingUnit(d_model)
        
    def forward(self, x):
                  
        residual = x
        
        dim0 = x.shape[0]
        dim1 = x.shape[1]
        dim2 = x.shape[2]
        x = x.reshape([dim0,dim1*dim2])
        x = self.norm_global(x)
      
        x = x.reshape([dim0,dim1,dim2])
        x = x + residual 
        
        
        residual = x
        
                  
        x = self.norm_local(x)
        x = self.gating(x)
                  
        out = x + residual
        
        
        return out



class Normalizer(nn.Module):
    def __init__(self, d_model,num_tokens, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[NormalizerBlock(d_model,num_tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








