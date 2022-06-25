import torch
from torch import nn

"""
This unit test was written as a sanity check that shuffle and unshuffle were inverse operations as they should be.
"""

downscale_ratio = 2

rearrange = nn.PixelUnshuffle(downscale_ratio)

#dummy = torch.Tensor(range(64)).reshape(2,2,4,4)
dummy = (torch.arange(4)+0.0).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat([1, int(downscale_ratio**2), 1, 1])
print("Dummy: ", dummy)

x = nn.PixelShuffle(downscale_ratio)(dummy)

print("Shuffled: ", x)
x = rearrange(x)

x = x.reshape(x.shape[0], -1, downscale_ratio**2, x.shape[-2], x.shape[-1])

x = torch.mean(x, dim=2)

print("Input:", dummy, "\nOutput:", x)
