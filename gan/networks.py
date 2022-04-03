import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), padding=padding)
        self.upscale_factor = upscale_factor
        self.rearrange = nn.PixelShuffle(upscale_factor)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling.
        # 1. Duplicate x channel wise upscale_factor^2 times.
        # 2. Then re-arrange to form an image of shape (batch x channel x height*upscale_factor x width*upscale_factor).
        # 3. Apply convolution.
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle
        # print(x.shape, "If this is B x C x H x W")
        x = x.repeat([1, int(self.upscale_factor**2), 1, 1])
        # print(x.shape, "Then this should be B x C * r^2 x H x W")
        x = self.rearrange(x)
        # print(x.shape, "Then this should be B x C x H*r x W*r")
        x = self.conv(x)
        # pass
        return x

class DownSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), padding=padding)
        self.downscale_ratio = downscale_ratio
        # print(downscale_ratio)
        self.rearrange = nn.PixelUnshuffle(downscale_ratio)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling.
        # 1. Re-arrange to form an image of shape: (batch x channel * upscale_factor^2 x height x width).
        # 2. Then split channel wise into upscale_factor^2 number of images of shape: (batch x channel x height x width).
        # 3. Average the images into one and apply convolution.
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle
        # print(x.shape, "Then this should be B x C x H*r x W*r")
        x = self.rearrange(x)
        # print(x.shape, "Then this should be B x C * r^2 x H x W")
        x = x.reshape(x.shape[0], -1, int(self.downscale_ratio**2), x.shape[-2], x.shape[-1])
        x  = torch.mean(x, dim=2)
        # print(x.shape, "If this is B x C x H x W")
        x = self.conv(x)
        # pass
        return x

class ResBlockUp(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        self.residual = UpSampleConv2D(n_filters, kernel_size, n_filters, padding=1)

        self.shortcut = UpSampleConv2D(input_channels, kernel_size=1, n_filters=n_filters)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        skip = self.shortcut(x)
        x = self.layers(x)
        x = self.residual(x)
        return x + skip


class ResBlockDown(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
        )
        (residual): DownSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): DownSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), padding=1),
            nn.ReLU()
        )
        self.residual = DownSampleConv2D(n_filters, kernel_size, n_filters, padding=1)
        self.shortcut = DownSampleConv2D(input_channels, kernel_size-2, n_filters)
    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        # pass
        skip = self.shortcut(x)
        x = self.layers(x)
        x = self.residual(x)
        # print("dimension x is: ", x.shape)
        # print("dimension skip is: ", skip.shape)

        return x + skip

class ResBlock(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size, kernel_size), padding=1),

        )
    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        # pass
        res = self.layers(x)
        return x + res

class Generator(jit.ScriptModule):
    # TODO 1.1: Imlpement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.dense = nn.Linear(128, 2048)
        self.starting_image_size = starting_image_size
        self.layers = nn.Sequential(
            ResBlockUp(128),
            ResBlockUp(128),
            ResBlockUp(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=(3,3), padding=1),
            nn.Tanh(),
        )

    @jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        # pass
        x = self.dense(z)
        x = x.reshape(x.shape[0], -1, self.starting_image_size, self.starting_image_size)
        return self.layers(x)
    @jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)
        # pass
        z = torch.normal(torch.zeros((n_samples, 128)), torch.ones((n_samples, 128))).half().cuda()
        x = self.forward_given_samples(z)
        return x

class Discriminator(jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            ResBlockDown(3),
            ResBlockDown(128),
            ResBlock(128),
            ResBlock(128),
            nn.ReLU(),
        )
        self.dense = nn.Linear(128, 1)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers and sum across the image dimensions before passing to the output layer!
        # pass
        # print("The image coming into the discriminator has a max and min of", torch.max(x) ,torch.min(x))
        x = self.layers(x)
        x = torch.sum(x, dim=[-2,-1])
        return self.dense(x)
