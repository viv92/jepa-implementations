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
from torchvision.utils import save_image, make_grid
import pickle 

# import T5 (we use the T5 Encoder only)
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from utils_dit_jepa_edm import *


# prediction function that predicts denoised x using the diffusion net, with input-output conditioning
# x: noised input 
# t: diffusion time = added noise sigma
def prediction_function(net, x, t, class_label, use_cnoise):
    c_noise = 0.25 * torch.log(t)
    # prop 
    if use_cnoise:
        t = c_noise 
    out = net(x, t, class_label)
    return out

# function to map discrete step n to continuous time t
# NOTE that in EDM paper, authors follow reverse step indexing mapping [N-1, 0] to time interval [t_eps, T] = [sigma_min, sigma_max]
# Also NOTE that in EDM paper, this function is used only during sampling
def step_to_time(rho, t_eps, T, N, n):
    inv_rho = 1/rho 
    a = math.pow(t_eps, inv_rho)
    b = math.pow(T, inv_rho)
    return torch.pow( b + ((a-b) * n)/(N-1), rho) 

def time_to_step(rho, t_eps, T, N, t):
    inv_rho = 1/rho 
    a = math.pow(t_eps, inv_rho)
    b = math.pow(T, inv_rho)
    return ( (N-1) * (torch.pow(t, inv_rho) - b) ) / (a-b)

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


def perturb_edm(x, t, mask_token, rho=7.0, t_eps=0.002, T=80, N=35):
    '''
    flip prob obtained as the CDF of an approximate integrable function of step_to_time for static hyperparams
    approx function: exp(-0.2 * n + 4.4)
    cdf: (exp(-0.2 * n + 4.4) - 0.0743) * (5/406.8)
    where n: step from time
    '''
    n = time_to_step(rho, t_eps, T, N, t)
    flip_prob = ( ( torch.exp(-0.2 * n + 4.4) - 0.0743 ) * 5 ) / 406.8
    flip_indices = torch.rand(*x.shape, device=x.device) < flip_prob.unsqueeze(-1)
    x_perturb = torch.where(flip_indices, mask_token, x) # fill the mask_token at flip_indices; fill the original token at other indices
    return x_perturb # x_perturb.shape: [b, seqlen]


# function to sample / generate img - deterministic sampling scheme using heun solver
# NOTE that step range [0 ... N-1, N] = time range [T ... t_eps, 0] = noise range [sigma_max ... sigma_min, 0]
def deterministic_sampling_heun_cfg(online_net, pred_net, use_cnoise, img_shape, rho, ts, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale, mask_token, device):

    # x_T
    x_n = mask_token * torch.ones(img_shape, dtype=torch.int64, device=device)

    t_n = ts[0]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]

    # get online latent
    if use_cnoise:
        t_n_input = 0.25 * torch.log(t_n)
    else:
        t_n_input = t_n 
    online_latent = online_net(x_n, t_n_input, class_label)
    pred_latent = prediction_function(pred_net, online_latent, t_n, class_label, use_cnoise)

    x_n = online_latent 
    t_n_xshape = expand_dims_to_match(t_n, x_n)

    for n in range(0, N): 

        # # perform inpainting
        # if x_inpaint is not None:

        #     z = torch.randn_like(x_inpaint)
        #     x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
        #     x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = prediction_function(pred_net, x_n, t_n, class_label, use_cnoise)
        d_n_cond = (x_n - pred_x_cond) / t_n_xshape 
        pred_x_uncond = prediction_function(pred_net, x_n, t_n, default_label, use_cnoise)
        d_n_uncond = (x_n - pred_x_uncond) / t_n_xshape 

        # cfg on score = cfg on predicted noise d_n
        d_n = cfg_scale * d_n_cond +  (1 - cfg_scale) * d_n_uncond

        x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = prediction_function(pred_net, x_n_plus1, t_n_plus1, class_label, use_cnoise)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = prediction_function(pred_net, x_n_plus1, t_n_plus1, default_label, use_cnoise)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

            x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n, online_latent, pred_latent


# function to sample / generate img - stochastic sampling scheme using heun solver
# NOTE that step range [0 ... N-1, N] = time range [T ... t_eps, 0] = noise range [sigma_max ... sigma_min, 0]
def stochastic_sampling_heun_cfg(online_net, pred_net, use_cnoise, img_shape, rho, ts, gammas, S_noise, class_label, default_label, x_inpaint, sigma_data, N, cfg_scale, start_time, mask_token, device):

    # x_T
    x_n = mask_token * torch.ones(img_shape, dtype=torch.int64, device=device)

    t_n = ts[0]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]

    # get online latent
    if use_cnoise:
        t_n_input = 0.25 * torch.log(t_n)
    else:
        t_n_input = t_n 
    online_latent = online_net(x_n, t_n_input, class_label)
    pred_latent = prediction_function(pred_net, online_latent, t_n, class_label, use_cnoise)

    x_n = online_latent 
    t_n_xshape = expand_dims_to_match(t_n, x_n)

    for n in range(0, N): 

        # # perform inpainting
        # if x_inpaint is not None:

        #     z = torch.randn_like(x_inpaint)
        #     x_inpaint_noised = x_inpaint + expand_dims_to_match(t_n, x_inpaint) * z
        #     x_n[:, :x_inpaint.shape[1]] = x_inpaint_noised


        z = torch.randn_like(x_n) * S_noise
        t_n_hat = t_n + gammas[n] * t_n
        t_n_hat_xshape = expand_dims_to_match(t_n_hat, x_n)
        t_delta = torch.sqrt( torch.pow(t_n_hat, 2) - torch.pow(t_n, 2) ) 
        x_n_hat = x_n + expand_dims_to_match(t_delta, x_n)  * z 

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = prediction_function(pred_net, x_n_hat, t_n_hat, class_label, use_cnoise)
        d_n_cond = (x_n_hat - pred_x_cond) / t_n_hat_xshape 
        pred_x_uncond = prediction_function(pred_net, x_n_hat, t_n_hat, default_label, use_cnoise)
        d_n_uncond = (x_n_hat - pred_x_uncond) / t_n_hat_xshape 

        # cfg on score = cfg on predicted noise d_n
        d_n = cfg_scale * d_n_cond + (1 - cfg_scale) * d_n_uncond

        x_n_plus1 = x_n_hat + (t_n_plus1_xshape - t_n_hat_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = prediction_function(pred_net, x_n_plus1, t_n_plus1, class_label, use_cnoise)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = prediction_function(pred_net, x_n_plus1, t_n_plus1, default_label, use_cnoise)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  

            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = cfg_scale * d_n_plus1_cond + (1 - cfg_scale) * d_n_plus1_uncond

            x_n_plus1 = x_n_hat + (t_n_plus1_xshape - t_n_hat_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n, online_latent, pred_latent


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
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                return model, optimizer, scheduler
            else:
                return model, optimizer
        else:
            return model 
        
# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer=None, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)


def ema(arr, val, r=0.01):
    if len(arr) == 0:
        return [val]
    newval = arr[-1] * (1-r) + val * r 
    arr.append(newval)
    return arr 


### main
if __name__ == '__main__':
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init program description / caption embedder model (codeT5)
    t5_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').to(device)
    t5_d_model = 768
    t5_vocab_size = t5_tokenizer.vocab_size # 32100
    t5_max_seq_len = t5_tokenizer.model_max_length # 512 # TODO this might require shortening the program descriptions
    # delete t5_decoder to save ram 
    del t5_model.decoder 
    # freeze t5_encoder
    freeze(t5_model.encoder)

    # hyperparams for vocab and bottleneck dims 
    dit_vocab_size = t5_vocab_size + 1
    dit_mask_token = dit_vocab_size - 1
    
    latent_dim = 8
    latent_seq_len = 64 
    tau = 0.05

    # hyperparams for custom decoder (DiT)
    d_model_dit = 512 
    n_heads_dit = 8
    assert d_model_dit % n_heads_dit == 0
    d_k_dit = d_model_dit // n_heads_dit 
    d_v_dit = d_k_dit 
    n_layers_dit = 6
    d_ff_dit = d_model_dit * 4

    d_model_pred_dit = 128 # 32
    assert d_model_pred_dit > latent_dim
    n_heads_pred_dit = 4 # 2
    assert d_model_pred_dit % n_heads_pred_dit == 0
    d_k_pred_dit = d_model_pred_dit // n_heads_pred_dit 
    d_v_pred_dit = d_k_pred_dit 
    n_layers_pred_dit = 3
    d_ff_pred_dit = d_model_pred_dit * 4

    dropout = 0.1
    weight_decay = 0.1 
    gradient_accumulation_steps = 4

    # hyperparams for consistency training
    start_time = 0.002 # start time t_eps of the ODE - the time interval is [t_eps, T] (continuous) and corresponding step interval is [1, N] (discrete)
    end_time = 80 # 16 # end time T of the ODE (decreasing end time leads to lower loss with some improvement in sample quality)
    N_final =  35 # final value of N in the step schedule (denoted as s_1 in appendix C)
    rho = 7.0 # used to calculate mapping from discrete step interval [1, N] to continuous time interval [t_eps, T]
    sigma_data = 0.5 # used to calculate c_skip and c_out to ensure boundary condition
    P_mean = -1.2 # mean of the train time noise sampling distribution (log-normal)
    P_std = 1.2 # std of the train time noise sampling distribution (log-normal)
    use_cnoise = False  
    S_noise = 1.003
    S_churn = 40
    S_tmin = 0.05
    S_tmax = 50

    sampling_strategy = 'deterministic'
    n_samples = 4

    total_train_steps = 10 ** 7
    train_steps_done = 0
    lr = 3e-4 # 1e-4 
    batch_size = 128
    random_seed = 10
    plot_freq = 600
    sample_freq = int(plot_freq / 10)
    model_save_freq = sample_freq
    p_uncond = 0.1 # for cfg
    cfg_scale = 2.5
    resume_training_from_ckpt = False       

    hyperparam_dict = {}
    hyperparam_dict['t0'] = start_time
    hyperparam_dict['tN'] = end_time
    hyperparam_dict['N_final'] = N_final
    hyperparam_dict['sampling'] = sampling_strategy
    hyperparam_dict['lr'] = lr
    hyperparam_dict['batch'] = batch_size
    hyperparam_dict['latDim'] = latent_dim
    hyperparam_dict['latSeq'] = latent_seq_len
    hyperparam_dict['tau'] = tau
    hyperparam_dict['D'] = d_model_dit
    hyperparam_dict['H'] = n_heads_dit
    hyperparam_dict['L'] = n_layers_dit
    hyperparam_dict['predD'] = d_model_pred_dit
    hyperparam_dict['predH'] = n_heads_pred_dit
    hyperparam_dict['predL'] = n_layers_pred_dit
    hyperparam_dict['dropout'] = dropout
    hyperparam_dict['Wdecay'] = weight_decay
    hyperparam_dict['gradAcc'] = gradient_accumulation_steps
    hyperparam_dict['pUncond'] = p_uncond
    hyperparam_dict['cfg'] = cfg_scale
    hyperparam_dict['useCnoise'] = use_cnoise 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + ':' + str(v) 

    save_folder = './results/jepa_edm_codeContests' + hyperparam_str
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save ckpt path
    online_net_ckpt_path = './ckpts/jepa_edm_codeContests_onlineNet' + hyperparam_str + '.pt'
    pred_net_ckpt_path = './ckpts/jepa_edm_codeContests_predNet' + hyperparam_str + '.pt'
    target_net_ckpt_path = './ckpts/jepa_edm_codeContests_targetNet' + hyperparam_str + '.pt'

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # load code contests dataset 
    train_dataset_path = '/home/vivswan/experiments/scalable_dreamcoder/filtered-dataset/filtered-hq-deduped-python.pkl'
    
    with open(train_dataset_path, 'rb') as f:
        train_dataset = pickle.load(f)

    # de-duplicate dataset based on problem statements
    seen_prompts = set() 
    deduped_dataset = []
    for x in train_dataset:
        if x['prompt'] not in seen_prompts:
            deduped_dataset.append(x)
            seen_prompts.add(x['prompt'])
    train_dataset = deduped_dataset
    print('len(deduped_dataset): ', len(train_dataset))
    np.random.shuffle(train_dataset)

    # init custom decoder (DiT)
    max_seq_len_dit = t5_max_seq_len + 1 # [t, x_noised]
    condition_dim = t5_d_model 
    
    online_net = init_jepa_dit_discrete(max_seq_len_dit, t5_max_seq_len, latent_seq_len, d_model_dit, dit_vocab_size, latent_dim, condition_dim, d_k_dit, d_v_dit, n_heads_dit, n_layers_dit, d_ff_dit, dropout, device).to(device)
    
    pred_net = init_jepa_dit(max_seq_len_dit, t5_max_seq_len, latent_seq_len, d_model_pred_dit, latent_dim, latent_dim, condition_dim, d_k_pred_dit, d_v_pred_dit, n_heads_pred_dit, n_layers_pred_dit, d_ff_pred_dit, dropout, device).to(device)

    target_net = deepcopy(online_net)

    # optimizer 
    optimizer_online_net = torch.optim.AdamW(params=online_net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    optimizer_pred_net = torch.optim.AdamW(params=pred_net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # load ckpt
    if resume_training_from_ckpt:
        online_net, optimizer_online_net = load_ckpt(online_net_ckpt_path, online_net, optimizer=optimizer_online_net, device=device, mode='train')
        pred_net, optimizer_pred_net = load_ckpt(pred_net_ckpt_path, pred_net, optimizer=optimizer_pred_net, device=device, mode='train')
        target_net = load_ckpt(target_net_ckpt_path, target_net, device=device, mode='train')

    # train

    train_step = train_steps_done
    epoch = 0
    results_train_loss, results_sample_loss, results_online_loss, results_pred_loss = [], [], [], []
    criterion = nn.MSELoss(reduction='none') # NOTE that reduction=None is necessary so that we can apply weighing factor lambda

    # calculate ts and gammas - NOTE these are used only for sampling in the EDM approach
    ts = calculate_ts(rho, start_time, end_time, N_final) 
    gammas = calculate_gammas(ts, S_churn, S_tmin, S_tmax)

    pbar = tqdm(total=total_train_steps)
    while train_step < total_train_steps + train_steps_done:

        # fetch train minibatch
        idx = np.arange(len(train_dataset))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [train_dataset[i] for i in idx]

        # prepare program tokens and caption (problem statement) embeddings
        with torch.no_grad():
            program_batch = [item['code'] for item in minibatch]
            program_tokens_dict = t5_tokenizer(program_batch, return_tensors='pt', padding='max_length', truncation=True)
            # TODO use obtained attention mask for fsq encoder 
            program_tokens, program_attn_mask = program_tokens_dict.input_ids, program_tokens_dict.attention_mask

            # get caption embeddings
            cap_batch = [item['prompt'] for item in minibatch]
            cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
            enc_out = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

        x = program_tokens.to(device)
        y = enc_out

        # for sampling 
        sample_caption_emb = y[:1] # NOTE that we sample one class label but generate n_sample imgs for that label
        sample_caption_emb = sample_caption_emb.expand(n_samples, -1, -1)

        # # for inpainting
        sample_xinpaint_emb = None
        # sample_xinpaint_emb = x[:1]
        # sample_xinpaint_emb = sample_xinpaint_emb[:, :int(sample_xinpaint_emb.shape[1]/2)] # pick first half for inpainting
        # sample_xinpaint_emb = sample_xinpaint_emb.expand(n_samples, -1, -1)

        # remove label with prob p_uncond
        if np.random.rand() < p_uncond: 
            y = None

        # get target latent 
        with torch.no_grad():
            if use_cnoise:
                t0 = torch.tensor(0.0001)
                t0 = 0.25 * torch.log(t0)
            else:
                t0 = torch.tensor(0.0)
            t0 = t0.unsqueeze(-1).expand(x.shape[0]).to(device)
            target_latent = target_net(x, t0, enc_out)

        # alternate way to sample time = sigma using change of variable (as used in EDM paper) 
        # NOTE that this directly gives the time t = sigma and not the step index n where t = ts[n]
        log_sigma = torch.randn(x.shape[0]) * P_std + P_mean 
        t_n = torch.exp(log_sigma).to(device)
        
        # perturb (mask) the data
        x_n = perturb_edm(x, t_n, dit_mask_token)

        # get online latent
        if use_cnoise:
            t_n_input = 0.25 * torch.log(t_n)
        else:
            t_n_input = t_n 
        online_latent = online_net(x_n, t_n_input, enc_out)

        # get pred latent
        pred_latent = prediction_function(pred_net, online_latent, t_n, y, use_cnoise)
        
        # calculate loss 
        weight_factor = calculate_lambda(t_n, sigma_data).to(device)
        weight_factor = expand_dims_to_match(weight_factor, target_latent)
        d = criterion(pred_latent, target_latent)
        loss = weight_factor * d 
        loss = loss.mean()

        # adjustment for gradient accumulation 
        loss_scaled = loss / gradient_accumulation_steps
        loss_scaled.backward()

        if (train_step + 1) % gradient_accumulation_steps == 0:
            # gradient cliping - helps to prevent unnecessary divergence 
            torch.nn.utils.clip_grad_norm_(pred_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
            # gradient step
            optimizer_pred_net.step()
            optimizer_online_net.step()
            optimizer_online_net.zero_grad()
            optimizer_pred_net.zero_grad()
            # update target net weights 
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

        # update loss (for plotting)
        results_train_loss = ema(results_train_loss, loss.item())

        train_step += 1
        pbar.update(1)
        pbar.set_description('loss:{:.3f}'.format(results_train_loss[-1]))

        # save ckpt 
        if train_step % model_save_freq == 0:
            save_ckpt(device, online_net_ckpt_path, online_net, optimizer_online_net)
            save_ckpt(device, pred_net_ckpt_path, pred_net, optimizer_pred_net)
            save_ckpt(device, target_net_ckpt_path, target_net)

        # sample
        if train_step % sample_freq == 0:
            
            online_net.eval()
            pred_net.eval()

            # sample points - equivalent to just evaluating the consistency function
            with torch.no_grad():

                default_caption_emb = None
                sample_shape = x[:n_samples].shape # since we want to sample 'n_sample' points

                if sampling_strategy == 'deterministic':
                    sampled_latent, online_latent, pred_latent = deterministic_sampling_heun_cfg(online_net, pred_net, use_cnoise, sample_shape, rho, ts, sample_caption_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, dit_mask_token, device)
                else:
                    sampled_latent, online_latent, pred_latent = stochastic_sampling_heun_cfg(online_net, pred_net, use_cnoise, sample_shape, rho, ts, gammas, S_noise, sample_caption_emb, default_caption_emb, sample_xinpaint_emb, sigma_data, N_final, cfg_scale, start_time, dit_mask_token, device)

                # get target latent and sample loss
                target_latent = target_latent[:1]
                target_latent = target_latent.expand(n_samples, -1, -1)
                sample_loss = criterion(sampled_latent, target_latent).mean()
                online_loss = criterion(online_latent, target_latent).mean()
                pred_loss = criterion(pred_latent, target_latent).mean()
                results_sample_loss.append(sample_loss.item())
                results_online_loss.append(online_loss.item())
                results_pred_loss.append(pred_loss.item())

            online_net.train()
            pred_net.train()

        if train_step % plot_freq == 0:

            # plot results
            fig, ax = plt.subplots(2,2, figsize=(15,10))

            ax[0,0].plot(results_train_loss, label='train_loss')
            ax[0,0].legend()
            ax[0,0].set(xlabel='train_iters')
            ax[0,0].set_title('val:{:.3f}'.format(results_train_loss[-1]))
            # ax[0,0].set_ylim([0, 2])

            ax[0,1].plot(results_sample_loss, label='sample_loss')
            ax[0,1].legend()
            ax[0,1].set(xlabel='eval_iters')
            ax[0,1].set_title('val:{:.3f}'.format(results_sample_loss[-1]))

            ax[1,0].plot(results_online_loss, label='online_loss')
            ax[1,0].legend()
            ax[1,0].set(xlabel='eval_iters')
            ax[1,0].set_title('val:{:.3f}'.format(results_online_loss[-1]))

            ax[1,1].plot(results_sample_loss, label='sample_loss')
            ax[1,1].plot(results_pred_loss, label='pred_loss')
            ax[1,1].plot(results_online_loss, label='online_loss')
            ax[1,1].legend()
            ax[1,1].set(xlabel='eval_iters')
            ax[1,1].set_title('sample:{:.3f} pred:{:.3f} online:{:.3f}'.format(results_sample_loss[-1], results_pred_loss[-1], results_online_loss[-1]))

            plt.savefig(save_folder + f'/loss_trainStep={train_step}.png' )



    pbar.close()
