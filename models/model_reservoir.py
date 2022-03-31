# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:37:32 2022

@author: Saber
"""

class Reservoir0(nn.Module):

    # A fully linear unit
    # Does not accept sequences, but does maintain a state
    
  def __init__(self, n_in, n_hidden=20):
    super().__init__()
    
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.state = torch.zeros((n_in, n_hidden))
    self.w_in = nn.Parameter(torch.rand(n_in, n_hidden))
    # self.w_out = nn.Parameter(torch.rand(n_in, n_hidden))
    
    # a fixed auxilliary tensor, used in every forward() call.
    self.ones = torch.ones_like(self.state)

  def init_state(self):
      self.state = torch.zeros((self.n_in, self.n_hidden))
      
  def forward(self, xin):
    assert len(xin.shape)==2
    # assumes xin is of shape (N, n_in)
    xin = torch.unsqueeze(xin, dim=-1)
    # xin: (N, n_in, 1)
    self.state = torch.mul(xin, self.w_in)+ \
                 torch.mul(self.state, (self.ones-self.w_in))
    # self.out = torch.mul(self.state, self.w_out).sum(dim=-1)
    return self.state

