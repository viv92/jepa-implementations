'''
### Program implementing a text-to-image model using EDM from Elucidating the Design Space of Diffusion Models Paper

## Features:
1. This is basically the Muse model, but we use a edm to predict the image latents
2. So the architecture is as follows:
2.1. The text (captions) are encoded using a pre-trained T5 encoder 
2.2. A pre-trained VQVAE is used to map between image pixel space and image latent space. An image is fed to VQVAE encoder to obtain the img latent
2.3. The edm is implemented by a T5 decoder (only this is trained, all other components are pre-trained and frozen)
2.4. The input to the T5 decoder is [time_emb, noised_img_latent, placeholder_for_denoised_img_latent]
2.5. The text embedding (obtained in 2.1 from T5 encoder) is fed as conditioning input to the T5 decoder via cross-attention
2.6. The denoised image latent (obtained from T5 decoder) are fed to VQVAE decoder to obtain the image

## Todos / Questions:
1. Classifier-free guidance
2. Does xattn based conditioning make sense or we need a purely causal decoder-only self attn based conditioning 
5. in dpct, note that time is diffusion time and not just positional int (as in case of LLM). So don't use sinusoidal embeddings for time
6. its important to have enough capacity in transformer backbone (d_model >= x_dim) 

'''

import os
import cv2
import math 
from copy import deepcopy 
from matplotlib import pyplot as plt 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

# import T5 (we use the T5 Encoder only)
from transformers import T5Tokenizer, T5ForConditionalGeneration
# import VQVAE for loading the pretrained weights
from fsq_transformer import FSQ_Transformer, init_transformer, patch_seq_to_img, img_to_patch_seq

from utils_dit_gendim_crossattn import *
from utils_dit_jepa_edm import *


def jepa_prediction_function(net, x, t, class_label, use_cnoise):
    c_noise = 0.25 * torch.log(t)
    # prop 
    if use_cnoise:
        t = c_noise 
    out = net(x, t, class_label)
    return out

def jepa_deterministic_sampling_heun_cfg(online_net, pred_net, use_cnoise, img_shape, rho, ts, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale):

    x_n = torch.randn(img_shape) * ts[0] # NOTE that initial noise x_0 ~ gaussian(0, T^2) and not x_0 ~ gaussian(0, identity)
    x_n = x_n.to(device)
    t_n = ts[0]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]
    t_n_xshape = expand_dims_to_match(t_n, x_n)

    # get online latent
    c_in = 1 / torch.sqrt( torch.pow(t_n, 2) + sigma_data ** 2 )
    c_in = expand_dims_to_match(c_in, x_n) 
    x_n_input = x_n * c_in
    if use_cnoise:
        t_n_input = 0.25 * torch.log(t_n)
    else:
        t_n_input = t_n 
    online_latent = online_net(x_n_input, t_n_input, class_label)
    pred_latent = jepa_prediction_function(pred_net, online_latent, t_n, class_label, use_cnoise)

    x_n = online_latent 

    for n in range(0, N): 

        # # perform inpainting
        # if x_inpaint is not None:

        #     z = torch.randn_like(x_inpaint)
        #     x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
        #     x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = jepa_prediction_function(pred_net, x_n, t_n, class_label, use_cnoise)
        d_n_cond = (x_n - pred_x_cond) / t_n_xshape 
        pred_x_uncond = jepa_prediction_function(pred_net, x_n, t_n, default_label, use_cnoise)
        d_n_uncond = (x_n - pred_x_uncond) / t_n_xshape 

        # cfg on score = cfg on predicted noise d_n
        d_n = cfg_scale * d_n_cond +  (1 - cfg_scale) * d_n_uncond

        x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = jepa_prediction_function(pred_net, x_n_plus1, t_n_plus1, class_label, use_cnoise)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = jepa_prediction_function(pred_net, x_n_plus1, t_n_plus1, default_label, use_cnoise)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

            x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n, online_latent, pred_latent


def jepa_cycle_deterministic_sampling_heun_cfg(online_net, pred_net, use_cnoise, img_shape, rho, ts, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale, start_x, start_n):

    t_n = ts[start_n]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]
    t_n_xshape = expand_dims_to_match(t_n, start_x)

    z = torch.randn_like(start_x)
    x_n = start_x + t_n_xshape * z 

    # get online latent
    c_in = 1 / torch.sqrt( torch.pow(t_n, 2) + sigma_data ** 2 )
    c_in = expand_dims_to_match(c_in, x_n) 
    x_n_input = x_n * c_in
    if use_cnoise:
        t_n_input = 0.25 * torch.log(t_n)
    else:
        t_n_input = t_n 
    online_latent = online_net(x_n_input, t_n_input, class_label)
    pred_latent = jepa_prediction_function(pred_net, online_latent, t_n, class_label, use_cnoise)

    x_n = online_latent 

    for n in range(start_n, N): 

        # # perform inpainting
        # if x_inpaint is not None:

        #     z = torch.randn_like(x_inpaint)
        #     x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
        #     x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = jepa_prediction_function(pred_net, x_n, t_n, class_label, use_cnoise)
        d_n_cond = (x_n - pred_x_cond) / t_n_xshape 
        pred_x_uncond = jepa_prediction_function(pred_net, x_n, t_n, default_label, use_cnoise)
        d_n_uncond = (x_n - pred_x_uncond) / t_n_xshape 

        # cfg on score = cfg on predicted noise d_n
        d_n = cfg_scale * d_n_cond +  (1 - cfg_scale) * d_n_uncond

        x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = jepa_prediction_function(pred_net, x_n_plus1, t_n_plus1, class_label, use_cnoise)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = jepa_prediction_function(pred_net, x_n_plus1, t_n_plus1, default_label, use_cnoise)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

            x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n, online_latent, pred_latent


# def jepa_stochastic_sampling_heun_cfg(online_net, pred_net, use_cnoise, img_shape, rho, ts, gammas, S_noise, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale, start_time, device):

#     x_n = torch.randn(img_shape) * ts[0] # NOTE that initial noise x_0 ~ gaussian(0, T^2) and not x_0 ~ gaussian(0, identity)
#     x_n = x_n.to(device)
#     t_n = ts[0]
#     t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]
#     t_n_xshape = expand_dims_to_match(t_n, x_n)

#     # get online latent
#     c_in = 1 / torch.sqrt( torch.pow(t_n, 2) + sigma_data ** 2 )
#     c_in = expand_dims_to_match(c_in, x_n) 
#     x_n_input = x_n * c_in
#     if use_cnoise:
#         t_n_input = 0.25 * torch.log(t_n)
#     else:
#         t_n_input = t_n 
#     online_latent = online_net(x_n_input, t_n_input, class_label)
#     pred_latent = jepa_prediction_function(pred_net, online_latent, t_n, class_label, use_cnoise)

#     x_n = online_latent 

#     for n in range(0, N): 

#         # # perform inpainting
#         # if x_inpaint is not None:

#         #     z = torch.randn_like(x_inpaint)
#         #     x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
#         #     x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised


#         z = torch.randn_like(x_n) * S_noise
#         t_n_hat = t_n + gammas[n] * t_n
#         t_n_hat_xshape = expand_dims_to_match(t_n_hat, x_n)
#         t_delta = torch.sqrt( torch.pow(t_n_hat, 2) - torch.pow(t_n, 2) ) 
#         x_n_hat = x_n + expand_dims_to_match(t_delta, x_n)  * z 

#         t_n_plus1 = ts[n+1]
#         t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
#         t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

#         pred_x_cond = jepa_prediction_function(pred_net, x_n_hat, t_n_hat, class_label, use_cnoise)
#         d_n_cond = (x_n_hat - pred_x_cond) / t_n_hat_xshape 
#         pred_x_uncond = jepa_prediction_function(pred_net, x_n_hat, t_n_hat, default_label, use_cnoise)
#         d_n_uncond = (x_n_hat - pred_x_uncond) / t_n_hat_xshape 

#         # cfg on score = cfg on predicted noise d_n
#         d_n = cfg_scale * d_n_cond + (1 - cfg_scale) * d_n_uncond

#         x_n_plus1 = x_n_hat + (t_n_plus1_xshape - t_n_hat_xshape) * d_n 

#         if not (n == N-1):

#             pred_x_plus1_cond = jepa_prediction_function(pred_net, x_n_plus1, t_n_plus1, class_label, use_cnoise)
#             d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
#             pred_x_plus1_uncond = jepa_prediction_function(pred_net, x_n_plus1, t_n_plus1, default_label, use_cnoise)
#             d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

#             # cfg on score = cfg on predicted noise d_n
#             d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

#             x_n_plus1 = x_n_hat + (t_n_plus1_xshape - t_n_hat_xshape) * (d_n + d_n_plus1) * 0.5

#         # for next iter 
#         x_n = x_n_plus1 
#         t_n = t_n_plus1 
#         t_n_xshape = t_n_plus1_xshape

#     return x_n, online_latent, pred_latent



# prediction function that predicts denoised x using the diffusion net, with input-output conditioning
# x: noised input 
# t: diffusion time = added noise sigma
def prediction_function(net, x, t, class_label, sigma_data, t_eps):
    # sigma_t = t - t_eps 
    sigma_t = t 
    c_in = 1 / torch.sqrt( torch.pow(sigma_t,2) + sigma_data ** 2 )
    c_skip = (sigma_data ** 2) / ( torch.pow(sigma_t, 2) + (sigma_data ** 2) )
    c_out = (sigma_data * sigma_t) / torch.sqrt( (sigma_data ** 2) + torch.pow(t, 2) )
    c_noise = 0.25 * torch.log(sigma_t)
    # expand dims
    c_in = expand_dims_to_match(c_in, x) 
    c_skip = expand_dims_to_match(c_skip, x)
    c_out = expand_dims_to_match(c_out, x)
    # prop 
    x_conditioned = c_in * x 
    if use_cnoise:
        t = c_noise 
    out = net(x_conditioned, t, class_label)
    out_conditioned = c_skip * x + c_out * out 
    return out_conditioned

# function to map discrete step n to continuous time t
# NOTE that in EDM paper, authors follow reverse step indexing mapping [N-1, 0] to time interval [t_eps, T] = [sigma_min, sigma_max]
# Also NOTE that in EDM paper, this function is used only during sampling
def step_to_time(rho, t_eps, T, N, n):
    inv_rho = 1/rho 
    a = math.pow(t_eps, inv_rho)
    b = math.pow(T, inv_rho)
    return torch.pow( b + ((a-b) * n)/(N-1), rho) 

# function to calculate the list of all time steps for the given schedule
# NOTE that in EDM paper, step interval [0 ... N-1, N] corresponds to time interval [T ... t_eps, 0]
def calculate_ts(rho, t_eps, T, N):
    ts = [] 
    for n in range(0, N):
        t_n = step_to_time(rho, t_eps, T, N, torch.tensor(n))
        ts.append(t_n)
    # append t[N] = 0
    ts.append(torch.tensor(0.0))
    return torch.tensor(ts) 

# function to calculate gammas - used for stochastic sampling
def calculate_gammas(ts, S_churn, S_tmin, S_tmax):
    N = len(ts)
    churn = torch.min( torch.tensor(S_churn/N), torch.tensor(math.sqrt(2)-1) )
    gammas = torch.zeros_like(ts)
    for i in range(N):
        if ts[i] > S_tmin and ts[i] < S_tmax:
            gammas[i] = churn 
    return gammas 

# function to calculate loss weight factor (lambda) 
def calculate_lambda(sigma_t, sigma_data):
    return ( torch.pow(sigma_t,2) + sigma_data ** 2 ) / torch.pow( sigma_t * sigma_data , 2)

# function to expand dims of tensor x to match that of tensor y 
def expand_dims_to_match(x, y):
    while len(x.shape) < len(y.shape):
        x = x.unsqueeze(-1)
    return x 


# function to sample / generate img - deterministic sampling scheme using heun solver
# NOTE that step range [0 ... N-1, N] = time range [T ... t_eps, 0] = noise range [sigma_max ... sigma_min, 0]
def deterministic_sampling_heun_cfg(net, img_shape, rho, ts, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale):

    x_n = torch.randn(img_shape) * ts[0] # NOTE that initial noise x_0 ~ gaussian(0, T^2) and not x_0 ~ gaussian(0, identity)
    x_n = x_n.to(device)
    t_n = ts[0]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]
    t_n_xshape = expand_dims_to_match(t_n, x_n)

    for n in range(0, N): 

        # perform inpainting
        if x_inpaint is not None:

            z = torch.randn_like(x_inpaint)
            x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
            x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = prediction_function(net, x_n, t_n, class_label, sigma_data, start_time)
        d_n_cond = (x_n - pred_x_cond) / t_n_xshape 
        pred_x_uncond = prediction_function(net, x_n, t_n, default_label, sigma_data, start_time)
        d_n_uncond = (x_n - pred_x_uncond) / t_n_xshape 

        # cfg on score = cfg on predicted noise d_n
        d_n = cfg_scale * d_n_cond +  (1 - cfg_scale) * d_n_uncond

        x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = prediction_function(net, x_n_plus1, t_n_plus1, class_label, sigma_data, start_time)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = prediction_function(net, x_n_plus1, t_n_plus1, default_label, sigma_data, start_time)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

            x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n 


# function to sample / generate img - stochastic sampling scheme using heun solver
# NOTE that step range [0 ... N-1, N] = time range [T ... t_eps, 0] = noise range [sigma_max ... sigma_min, 0]
def stochastic_sampling_heun_cfg(net, img_shape, rho, ts, gammas, S_noise, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale, start_time, device):

    x_n = torch.randn(img_shape) * ts[0] # NOTE that initial noise x_0 ~ gaussian(0, T^2) and not x_0 ~ gaussian(0, identity)
    x_n = x_n.to(device)
    t_n = ts[0]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]
    t_n_xshape = expand_dims_to_match(t_n, x_n)

    for n in range(0, N): 

        # perform inpainting
        if x_inpaint is not None:

            z = torch.randn_like(x_inpaint)
            x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
            x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised


        z = torch.randn_like(x_n) * S_noise
        t_n_hat = t_n + gammas[n] * t_n
        t_n_hat_xshape = expand_dims_to_match(t_n_hat, x_n)
        t_delta = torch.sqrt( torch.pow(t_n_hat, 2) - torch.pow(t_n, 2) ) 
        x_n_hat = x_n + expand_dims_to_match(t_delta, x_n)  * z 

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = prediction_function(net, x_n_hat, t_n_hat, class_label, sigma_data, start_time)
        d_n_cond = (x_n_hat - pred_x_cond) / t_n_hat_xshape 
        pred_x_uncond = prediction_function(net, x_n_hat, t_n_hat, default_label, sigma_data, start_time)
        d_n_uncond = (x_n_hat - pred_x_uncond) / t_n_hat_xshape 

        # cfg on score = cfg on predicted noise d_n
        d_n = cfg_scale * d_n_cond + (1 - cfg_scale) * d_n_uncond

        x_n_plus1 = x_n_hat + (t_n_plus1_xshape - t_n_hat_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = prediction_function(net, x_n_plus1, t_n_plus1, class_label, sigma_data, start_time)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = prediction_function(net, x_n_plus1, t_n_plus1, default_label, sigma_data, start_time)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

            x_n_plus1 = x_n_hat + (t_n_plus1_xshape - t_n_hat_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n 


# fetch dataset - using data loader
def mnist_dl(img_size, batch_size):
    tf = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(img_size),  # args.image_size 
        # torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5) # equivalent to transforming pixel values from range [0,1] to [-1,1]
    ])
    dataset = MNIST(
        "/home/vivswan/experiments/fsq/dataset_mnist",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return dataloader


# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False) 

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer
        
# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)
        

# convert tensor to img
def to_img(x):
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.clamp(0, 1) # clamp img to be strictly in [-1, 1]
    x = x.permute(0,2,3,1) # [b,c,h,w] -> [b,h,w,c]
    return x 

# function to save a generated img
def save_img_generated(x_g, save_path):
    gen_img = x_g.detach().cpu().numpy()
    gen_img = np.uint8( gen_img * 255 )
    # bgr to rgb 
    gen_img = gen_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)

def ema(arr, val, r=0.01):
    if len(arr) == 0:
        return [val]
    newval = arr[-1] * (1-r) + val * r 
    arr.append(newval)
    return arr 
        


### main
if __name__ == '__main__':
    # hyperparams for vqvae (FSQ_Transformer)
    img_size = 28 # mnist
    img_channels = 1
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)

    patch_size = 4 # as required by the pretrained FSQ_Transformer
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item

    # hyperparams for custom decoder (DiT)
    d_model_dit = 512 # latent_dim * 64
    n_heads_dit = 2
    assert d_model_dit % n_heads_dit == 0
    d_k_dit = d_model_dit // n_heads_dit 
    d_v_dit = d_k_dit 
    n_layers_dit = 6
    d_ff_dit = d_model_dit * 4
    dropout = 0.1

    d_model_t5 = 768 # d_model for T5 (required for image latents projection)
    max_seq_len_t5 = 512 # required to init T5 Tokenizer


    # hyperparams for jepa
    jepa_patch_dim = patch_dim 
    jepa_seq_len = seq_len 
    jepa_latent_dim = 4 # 64
    jepa_latent_seq_len = 8 # seq_len
    jepa_use_cnoise = False 
    
    jepa_d_model_dit = 512 # latent_dim * 64
    jepa_n_heads_dit = 8 # 2
    assert jepa_d_model_dit % jepa_n_heads_dit == 0
    jepa_d_k_dit = jepa_d_model_dit // jepa_n_heads_dit 
    jepa_d_v_dit = jepa_d_k_dit 
    jepa_n_layers_dit = 3 # 6
    jepa_d_ff_dit = jepa_d_model_dit * 4

    jepa_d_model_pred_dit = 32 # 128
    assert jepa_d_model_pred_dit > jepa_latent_dim
    jepa_n_heads_pred_dit = 2 # 4
    assert jepa_d_model_pred_dit % jepa_n_heads_pred_dit == 0
    jepa_d_k_pred_dit = jepa_d_model_pred_dit // jepa_n_heads_pred_dit 
    jepa_d_v_pred_dit = jepa_d_k_pred_dit 
    jepa_n_layers_pred_dit = 2
    jepa_d_ff_pred_dit = jepa_d_model_pred_dit * 4


    # hyperparams for consistency training
    start_time = 0.002 # start time t_eps of the ODE - the time interval is [t_eps, T] (continuous) and corresponding step interval is [1, N] (discrete)
    end_time = 80 # 16 # end time T of the ODE (decreasing end time leads to lower loss with some improvement in sample quality)
    N_final =  35 # final value of N in the step schedule (denoted as s_1 in appendix C)
    rho = 7.0 # used to calculate mapping from discrete step interval [1, N] to continuous time interval [t_eps, T]
    sigma_data = 0.5 # used to calculate c_skip and c_out to ensure boundary condition
    P_mean = -1.2 # mean of the train time noise sampling distribution (log-normal)
    P_std = 1.2 # std of the train time noise sampling distribution (log-normal)
    use_cnoise = True  
    S_noise = 1.003
    S_churn = 40
    S_tmin = 0.05
    S_tmax = 50

    sampling_strategy = 'stochastic' # 'deterministic'
    n_samples = 16 

    num_train_steps_per_epoch = 118 
    num_epochs = 100
    total_train_steps = num_train_steps_per_epoch * num_epochs
    train_steps_done = 28617
    lr = 3e-4 # 1e-4 
    batch_size = 512 # lower batch size allows for more training steps per diffusion process (but reduces compute efficiency)
    random_seed = 101010
    sample_freq = int(total_train_steps / 60)
    model_save_freq = int(total_train_steps / 10)
    plot_freq = model_save_freq
    p_uncond = 0.1 # for cfg
    cfg_scale = 2.5
    resume_training_from_ckpt = True        

    hyperparam_dict = {}
    hyperparam_dict['t0'] = start_time
    hyperparam_dict['tN'] = end_time
    hyperparam_dict['N_final'] = N_final
    hyperparam_dict['sampleStrat'] = sampling_strategy
    hyperparam_dict['lr'] = lr
    hyperparam_dict['B'] = batch_size
    hyperparam_dict['D'] = d_model_dit
    hyperparam_dict['H'] = n_heads_dit
    hyperparam_dict['L'] = n_layers_dit
    hyperparam_dict['jLatD'] = jepa_latent_dim
    hyperparam_dict['jLatSeq'] = jepa_latent_seq_len
    hyperparam_dict['jPredD'] = jepa_d_model_pred_dit
    hyperparam_dict['jPredH'] = jepa_n_heads_pred_dit
    hyperparam_dict['dropout'] = dropout
    hyperparam_dict['pUncond'] = p_uncond
    hyperparam_dict['cfgScale'] = cfg_scale
    hyperparam_dict['useCnoise'] = use_cnoise 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + ':' + str(v) 

    save_folder = './results/rhcdm_cfg_mnist' + hyperparam_str
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # muse_vqvae_edm save ckpt path
    muse_ckpt_path = './ckpts/rhcdm_cfg_mnist' + hyperparam_str + '.pt'

    # pretrained jepa edm ckpt path
    jepa_edm_onlineNet_path = './ckpts/jepa_edm_mnist_onlineNet|t0:0.002|tN:80|N_final:35|sampleStrat:deterministic|lr:0.0003|batch:512|latentDim:4|latentSeq:8|tau:0.05|D:512|H:8|L:3|predD:32|predH:2|predL:2|dropout:0.1|pUncond:0.1|cfgScale:2.5|useCnoise:False.pt' 
    jepa_edm_predNet_path = './ckpts/jepa_edm_mnist_predNet|t0:0.002|tN:80|N_final:35|sampleStrat:deterministic|lr:0.0003|batch:512|latentDim:4|latentSeq:8|tau:0.05|D:512|H:8|L:3|predD:32|predH:2|predL:2|dropout:0.1|pUncond:0.1|cfgScale:2.5|useCnoise:False.pt' 
    jepa_edm_targetNet_path = './ckpts/jepa_edm_mnist_targetNet|t0:0.002|tN:80|N_final:35|sampleStrat:deterministic|lr:0.0003|batch:512|latentDim:4|latentSeq:8|tau:0.05|D:512|H:8|L:3|predD:32|predH:2|predL:2|dropout:0.1|pUncond:0.1|cfgScale:2.5|useCnoise:False.pt' 

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load img dataset
    dataloader = mnist_dl(img_size, batch_size)

    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=max_seq_len_t5)
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

    # delete t5_decoder to save ram 
    del t5_model.decoder 

    # init custom decoder (DiT)
    max_seq_len_dit = seq_len + 1 # [t, x_noised]
    condition_dim = jepa_latent_dim # d_model_t5 # 1 # mnist label 
    net = init_dit(max_seq_len_dit, seq_len, d_model_dit, patch_dim, condition_dim, d_k_dit, d_v_dit, n_heads_dit, n_layers_dit, d_ff_dit, dropout, device).to(device)

    # freeze vqvae, t5_encoder and ema_net
    freeze(t5_model.encoder)

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr)

    # load ckpt
    if resume_training_from_ckpt:
        net, optimizer = load_ckpt(muse_ckpt_path, net, optimizer, device=device, mode='train')

    # init and load pretrained jepa models 
    jepa_max_seq_len_dit = jepa_seq_len + 1
    jepa_condition_dim = d_model_t5
    jepa_online_net = init_jepa_dit(jepa_max_seq_len_dit, jepa_seq_len, jepa_latent_seq_len, jepa_d_model_dit, jepa_patch_dim, jepa_latent_dim, jepa_condition_dim, jepa_d_k_dit, jepa_d_v_dit, jepa_n_heads_dit, jepa_n_layers_dit, jepa_d_ff_dit, dropout, device).to(device)
    
    jepa_pred_net = init_jepa_dit(jepa_max_seq_len_dit, jepa_seq_len, jepa_latent_seq_len, jepa_d_model_pred_dit, jepa_latent_dim, jepa_latent_dim, jepa_condition_dim, jepa_d_k_pred_dit, jepa_d_v_pred_dit, jepa_n_heads_pred_dit, jepa_n_layers_pred_dit, jepa_d_ff_pred_dit, dropout, device).to(device)

    jepa_target_net = deepcopy(jepa_online_net)

    jepa_online_net = load_ckpt(jepa_edm_onlineNet_path, jepa_online_net, device=device, mode='eval')
    jepa_pred_net = load_ckpt(jepa_edm_predNet_path, jepa_pred_net, device=device, mode='eval')
    jepa_target_net = load_ckpt(jepa_edm_targetNet_path, jepa_target_net, device=device, mode='eval')

    # train

    train_step = train_steps_done
    epoch = 0
    results_edm_loss, results_jepa_online_loss, results_jepa_pred_loss = [], [], []
    criterion = nn.MSELoss(reduction='none') # NOTE that reduction=None is necessary so that we can apply weighing factor lambda

    # calculate ts and gammas - NOTE these are used only for sampling in the EDM approach
    ts = calculate_ts(rho, start_time, end_time, N_final) 
    gammas = calculate_gammas(ts, S_churn, S_tmin, S_tmax)

    pbar = tqdm(total=num_epochs)
    while epoch < num_epochs:

        # fetch minibatch
        pbar2 = tqdm(dataloader)
        for imgs, labels in pbar2:

            imgs = imgs.to(device)

            # tokenize labels
            cap_list = labels.tolist()
            cap_list = [str(x) for x in cap_list]
            cap_tokens_dict = t5_tokenizer(cap_list, return_tensors='pt', padding=True, truncation=True)
            cap_tokens_dict = cap_tokens_dict.to(device)

            with torch.no_grad():
                # convert img to sequence of patches
                x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]

                # extract cap tokens and attn_mask from cap_tokens_dict
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                # feed cap_tokens to t5 encoder to get encoder output
                cap_enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

                # get jepa encoding 
                if jepa_use_cnoise:
                    t0 = torch.tensor(0.0001)
                    t0 = 0.25 * torch.log(t0)
                else:
                    t0 = torch.tensor(0.0) # TODO might want to use a less strict value eg ts[-2]
                t0 = t0.unsqueeze(-1).expand(x.shape[0]).to(device)
                online_latent = jepa_online_net(x, t0, cap_enc_out)
                pred_latent = jepa_prediction_function(jepa_pred_net, online_latent, t0, cap_enc_out, jepa_use_cnoise)

                target_latent = jepa_target_net(x, t0, cap_enc_out)
                jepa_online_loss = criterion(online_latent, target_latent).mean()
                jepa_pred_loss = criterion(pred_latent, target_latent).mean()


            y = pred_latent 

            # for sampling 
            sample_jepa_pred_emb = y[:1] # NOTE that we sample one class label but generate n_sample imgs for that label
            sample_jepa_pred_emb = sample_jepa_pred_emb.expand(n_samples, -1, -1)
            sample_caption_emb = cap_enc_out[:1]
            sample_caption_emb = sample_caption_emb.expand(n_samples, -1, -1)
            sample_x = x[:1]
            sample_x = sample_x.expand(n_samples, -1, -1)

            # # for inpainting
            sample_xinpaint_emb = None
            # sample_xinpaint_emb = x[:1]
            # sample_xinpaint_emb = sample_xinpaint_emb[:, :int(sample_xinpaint_emb.shape[1]/2)] # pick first half for inpainting
            # sample_xinpaint_emb = sample_xinpaint_emb.expand(n_samples, -1, -1)

            # remove label with prob p_uncond
            if np.random.rand() < p_uncond: 
                y = None

            # alternate way to sample time = sigma using change of variable (as used in EDM paper) 
            # NOTE that this directly gives the time t = sigma and not the step index n where t = ts[n]
            log_sigma = torch.randn(x.shape[0]) * P_std + P_mean 
            t_n = torch.exp(log_sigma).to(device)
            
            # get corresponding noised data points
            z = torch.randn_like(x)
            x_n = x + expand_dims_to_match(t_n, x) * z 
            # predict x_0
            pred_x = prediction_function(net, x_n, t_n, y, sigma_data, start_time) # pred_x.shape: [b, x_seq_len, patch_dim]
            
            # calculate loss 
            weight_factor = calculate_lambda(t_n, sigma_data).to(device)
            weight_factor = expand_dims_to_match(weight_factor, x)
            d = criterion(pred_x, x)
            loss = weight_factor * d 
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update losses (for plotting)
            results_edm_loss = ema(results_edm_loss, loss.item())
            results_jepa_online_loss = ema(results_jepa_online_loss, jepa_online_loss.item())
            results_jepa_pred_loss = ema(results_jepa_pred_loss, jepa_pred_loss.item())

            train_step += 1
            pbar2.update(1)
            pbar2.set_description('loss:{:.3f}'.format(results_edm_loss[-1]))

            # save ckpt 
            if train_step % model_save_freq == 0:
                save_ckpt(device, muse_ckpt_path, net, optimizer)

            # sample
            if train_step % sample_freq == 0:
                
                net.eval()

                with torch.no_grad():

                    sample_caption_string = cap_list[0]
                    default_caption_emb = None
                    sample_shape = sample_x.shape # since we want to sample 'n_sample' points

                    ## sample conditioned on jepa_pred_emb

                    if sampling_strategy == 'deterministic':
                        gen_img_patch_seq = deterministic_sampling_heun_cfg(net, sample_shape, rho, ts, sample_jepa_pred_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale)
                    else:
                        gen_img_patch_seq = stochastic_sampling_heun_cfg(net, sample_shape, rho, ts, gammas, S_noise, sample_jepa_pred_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_time, device)

                    # convert patch sequence to img 
                    gen_imgs_jepa_pred_emb = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]


                    ## sample conditioned on generated jepa_pred_emb

                    # deterministic
                    sampled_latent, online_latent, pred_latent = jepa_deterministic_sampling_heun_cfg(jepa_online_net, jepa_pred_net, jepa_use_cnoise, sample_shape, rho, ts, sample_caption_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale)

                    if sampling_strategy == 'deterministic':
                        gen_img_patch_seq = deterministic_sampling_heun_cfg(net, sample_shape, rho, ts, sampled_latent, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale)
                    else:
                        gen_img_patch_seq = stochastic_sampling_heun_cfg(net, sample_shape, rho, ts, gammas, S_noise, sampled_latent, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_time, device)

                    # convert patch sequence to img 
                    gen_imgs_jepa_sampled_emb = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]


                    # cycled 
                    n_cycles = 4 
                    start_n = int(N_final / 2) # / 4
                    for i in range(n_cycles):
                        sampled_latent_cycled, online_latent, pred_latent = jepa_cycle_deterministic_sampling_heun_cfg(jepa_online_net, jepa_pred_net, jepa_use_cnoise, sample_shape, rho, ts, sample_caption_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_x=gen_img_patch_seq, start_n=start_n)

                    if sampling_strategy == 'deterministic':
                        gen_img_patch_seq = deterministic_sampling_heun_cfg(net, sample_shape, rho, ts, sampled_latent_cycled, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale)
                    else:
                        gen_img_patch_seq = stochastic_sampling_heun_cfg(net, sample_shape, rho, ts, gammas, S_noise, sampled_latent_cycled, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_time, device)

                    # convert patch sequence to img 
                    gen_imgs_jepa_sampled_cycled_emb = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]


                    # # stochastic
                    # sampled_stochastic_latent, online_latent, pred_latent = jepa_stochastic_sampling_heun_cfg(jepa_online_net, jepa_pred_net, use_cnoise, sample_shape, rho, ts, gammas, S_noise, sample_caption_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_time, device)

                    # if sampling_strategy == 'deterministic':
                    #     gen_img_patch_seq = deterministic_sampling_heun_cfg(net, sample_shape, rho, ts, sampled_stochastic_latent, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale)
                    # else:
                    #     gen_img_patch_seq = stochastic_sampling_heun_cfg(net, sample_shape, rho, ts, gammas, S_noise, sampled_stochastic_latent, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_time, device)

                    # # convert patch sequence to img 
                    # gen_imgs_jepa_sampled_stochastic_emb = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]


                    # convert patch sequence to img 
                    sample_x = patch_seq_to_img(sample_x, patch_size, img_channels) # [b,c,h,w]

                    sample_x = make_grid(sample_x, nrow=4)
                    save_image(sample_x, f"{save_folder}/original_trainStep={train_step}_caption={sample_caption_string}.png")

                    gen_imgs_jepa_pred_emb = (gen_imgs_jepa_pred_emb * 0.5 + 0.5).clamp(0,1)
                    gen_imgs_jepa_pred_emb = make_grid(gen_imgs_jepa_pred_emb, nrow=4)
                    save_image(gen_imgs_jepa_pred_emb, f"{save_folder}/jepa_pred_trainStep={train_step}_caption={sample_caption_string}.png")

                    gen_imgs_jepa_sampled_emb = (gen_imgs_jepa_sampled_emb * 0.5 + 0.5).clamp(0,1)
                    gen_imgs_jepa_sampled_emb = make_grid(gen_imgs_jepa_sampled_emb, nrow=4)
                    save_image(gen_imgs_jepa_sampled_emb, f"{save_folder}/jepa_sampled_trainStep={train_step}_caption={sample_caption_string}.png")

                    gen_imgs_jepa_sampled_cycled_emb = (gen_imgs_jepa_sampled_cycled_emb * 0.5 + 0.5).clamp(0,1)
                    gen_imgs_jepa_sampled_cycled_emb = make_grid(gen_imgs_jepa_sampled_cycled_emb, nrow=4)
                    save_image(gen_imgs_jepa_sampled_cycled_emb, f"{save_folder}/jepa_sampled_cycled4_trainStep={train_step}_caption={sample_caption_string}.png")

                    # gen_imgs_jepa_sampled_stochastic_emb = (gen_imgs_jepa_sampled_stochastic_emb * 0.5 + 0.5).clamp(0,1)
                    # gen_imgs_jepa_sampled_stochastic_emb = make_grid(gen_imgs_jepa_sampled_stochastic_emb, nrow=4)
                    # save_image(gen_imgs_jepa_sampled_stochastic_emb, f"{save_folder}/jepa_sampled_stochastic_trainStep={train_step}_caption={sample_caption_string}.png")

                net.train()

            if train_step % plot_freq == 0:

                # plot results
                fig, ax = plt.subplots(1,2, figsize=(15,10))

                ax[0].plot(results_edm_loss, label='train_loss')
                ax[0].legend()
                ax[0].set(xlabel='train_iters')
                ax[0].set_title('val:{:.3f}'.format(results_edm_loss[-1]))
                # ax[0].set_ylim([0, 2])

                ax[1].plot(results_jepa_online_loss, label='jepa_online_loss')
                ax[1].plot(results_jepa_pred_loss, label='jepa_pred_loss')
                ax[1].legend()
                ax[1].set(xlabel='train_iters')
                ax[1].set_title('online:{:.3f} pred:{:.3f}'.format(results_jepa_online_loss[-1], results_jepa_pred_loss[-1]))

                plt.savefig(save_folder + f'/loss_trainStep={train_step}.png' )

        epoch += 1
        pbar.update(1)
        pbar2.close()


    pbar.close()
