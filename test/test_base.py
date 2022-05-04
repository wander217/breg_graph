import numpy as np
import torch

a = np.zeros((2,))
print(np.concatenate([a, [1, 2]], axis=-1).shape)

a, b = map(list, zip([1, 2], [3, 4]))
print(a, b)

print(torch.cat([torch.zeros((3, 1)), torch.zeros((4, 1))]))
