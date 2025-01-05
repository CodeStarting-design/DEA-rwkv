import torch
from model import *
from thop import profile


net  = DEANet(base_dim=32)  # 定义好的网络模型


input = torch.randn(3, 3, 256, 256)
flops, params = profile(net, (input,))
print('flops: G', flops/1e9, 'params: M', params/1e6)