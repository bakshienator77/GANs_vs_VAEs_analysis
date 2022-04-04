import os

import torch

from networks import Discriminator, Generator
from train import train_model
import torch.nn.functional as F



def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    grad_params = torch.autograd.grad(discrim_interp, interp, torch.ones_like(discrim_interp), 
                                create_graph=True, retain_graph=True)[0]
    # loss = torch.mean(torch.sigmoid(discrim_fake)) - torch.mean(torch.sigmoid(discrim_real))  #+ lamb * torch.mean((torch.norm(grad.reshape(grad.shape[0], -1), dim=1) - 1.0)**2)
    loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) + F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    # print("Grad params shape : ", grad_params.shape)
    # grad = grad_params.reshape(grad_params.shape[0], -1)
    # grad_norm = torch.sqrt(torch.sum(grad**2, dim=1) + 1e-12)
    # print("Norm Grad: ", torch.norm(grad.reshape(grad.shape[0], -1), dim=1))
    # print("Norm Grad shape: ", torch.norm(grad.reshape(grad.shape[0], -1), dim=1).shape)
    # print("Grad shape: ", grad.shape)
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()

    # print("grad norm: ", grad_norm.item())
    # print("grad norm shape: ", grad_norm.shape)
    # print("gradient penalty: ", torch.mean((grad_norm - 1.0)**2))
    loss = loss + lamb* torch.mean((grad_norm - 1.0)**2)
    return loss


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    # loss = - torch.mean(torch.sigmoid(discrim_fake))
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
