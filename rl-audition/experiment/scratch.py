import torch

a = torch.tensor([5.2, 0, 3.5])
b = torch.tensor([0, 0, 7.3])

print(torch.atan2(a, b))