import os
import math 
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np
import torch
torch.set_float32_matmul_precision('high') # use TF32 precision for speeding up matmul
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from time import time 
import random 
import pickle 

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from utils_dit_jepa_edm import *
from utils_dit_gendim_crossattn_sedd import *


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


def ema(arr, val, r=0.01):
    if len(arr) == 0:
        return [val]
    newval = arr[-1] * (1-r) + val * r 
    arr.append(newval)
    return arr 

# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


## main 
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

    # hyperparams for dit (jepa)
    jepa_latent_dim = 8
    jepa_latent_seq_len = 64 
    
    jepa_d_model_dit = 512 
    jepa_n_heads_dit = 8
    assert jepa_d_model_dit % jepa_n_heads_dit == 0
    jepa_d_k_dit = jepa_d_model_dit // jepa_n_heads_dit 
    jepa_d_v_dit = jepa_d_k_dit 
    jepa_n_layers_dit = 6
    jepa_d_ff_dit = jepa_d_model_dit * 4

    jepa_d_model_pred_dit = 32 # 128
    assert jepa_d_model_pred_dit > jepa_latent_dim
    jepa_n_heads_pred_dit = 2 # 4
    assert jepa_d_model_pred_dit % jepa_n_heads_pred_dit == 0
    jepa_d_k_pred_dit = jepa_d_model_pred_dit // jepa_n_heads_pred_dit 
    jepa_d_v_pred_dit = jepa_d_k_pred_dit 
    jepa_n_layers_pred_dit = 3
    jepa_d_ff_pred_dit = jepa_d_model_pred_dit * 4

    # hyperparams for edm (jepa)
    start_time = 0.002 # start time t_eps of the ODE - the time interval is [t_eps, T] (continuous) and corresponding step interval is [1, N] (discrete)
    end_time = 80 # 16 # end time T of the ODE (decreasing end time leads to lower loss with some improvement in sample quality)
    N_final =  35 # final value of N in the step schedule (denoted as s_1 in appendix C)
    rho = 7.0 # used to calculate mapping from discrete step interval [1, N] to continuous time interval [t_eps, T]
    sigma_data = 0.5 # used to calculate c_skip and c_out to ensure boundary condition
    P_mean = -1.2 # mean of the train time noise sampling distribution (log-normal)
    P_std = 1.2 # std of the train time noise sampling distribution (log-normal)
    jepa_use_cnoise = False  
    S_noise = 1.003
    S_churn = 40
    S_tmin = 0.05
    S_tmax = 50
    n_samples = 16 # 4

    # hyperparams for dit (sedd)
    dit_d_model = 512 
    dit_n_heads = 8
    assert dit_d_model % dit_n_heads == 0
    dit_d_k = dit_d_model // dit_n_heads
    dit_d_v = dit_d_k 
    dit_n_layers = 6 
    dit_d_ff = dit_d_model * 4

    # hyperparams for sedd 
    diffusion_start_time_eps = 1e-3
    num_sampling_steps = int(t5_max_seq_len * 0.25)

    # hyperparams for training (common)

    dropout = 0.1 
    weight_decay = 0.01 # 0.1 
    
    batch_size = 32
    gradient_accumulation_steps = 8 
    lr = 1e-4 # 3e-4
    num_train_steps = 10 ** 7
    train_steps_done = 0
    random_seed = 10
    resume_training_from_ckpt = False      
    train_dataset_size = 1000

    p_uncond = 0.1 
    cfg_scale = 2.5 # 2.0
    sampling_eps = 0 # 0.1 # 0.3       

    # hyperparams for figures and plotting
    plot_freq = 10 ** 4
    sampling_freq = int(plot_freq / 4)    
    
    # create hyperparam str
    hyperparam_dict = {}
    hyperparam_dict['method'] = 'rhcdm_cfg_codeContests' 
    hyperparam_dict['lr'] = lr
    hyperparam_dict['B'] = batch_size
    hyperparam_dict['D'] = dit_d_model
    hyperparam_dict['H'] = dit_n_heads
    hyperparam_dict['L'] = dit_n_layers
    hyperparam_dict['jLatD'] = jepa_latent_dim
    hyperparam_dict['jLatSeq'] = jepa_latent_seq_len
    hyperparam_dict['jPredD'] = jepa_d_model_pred_dit
    hyperparam_dict['jPredH'] = jepa_n_heads_pred_dit
    hyperparam_dict['Wdecay'] = weight_decay
    # hyperparam_dict['drop'] = dropout
    hyperparam_dict['gradAcc'] = gradient_accumulation_steps
    hyperparam_dict['nSamp'] = n_samples
    hyperparam_dict['sampSteps'] = num_sampling_steps
    hyperparam_dict['pUncond'] = p_uncond
    hyperparam_dict['cfg'] = cfg_scale
    hyperparam_dict['sampEps'] = sampling_eps
    hyperparam_dict['dataSz'] = train_dataset_size

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + '=' + str(v)

    results_dir = './results/' + hyperparam_str + '/'
    ckpts_dir = './ckpts/'
      
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    # dit (sedd) ckpt path 
    dit_ckpt_path = ckpts_dir + hyperparam_str + '.pt'

    # pretrained jepa edm ckpt path
    jepa_edm_onlineNet_path = './ckpts/jepa_edm_codeContests_onlineNet|t0:0.002|tN:80|N_final:35|sampling:deterministic|lr:0.0003|batch:128|latDim:8|latSeq:64|tau:0.05|D:512|H:8|L:6|predD:32|predH:2|predL:3|dropout:0.1|Wdecay:0.1|gradAcc:4|pUncond:0.1|cfg:2.5|useCnoise:False.pt' 
    jepa_edm_predNet_path = './ckpts/jepa_edm_codeContests_predNet|t0:0.002|tN:80|N_final:35|sampling:deterministic|lr:0.0003|batch:128|latDim:8|latSeq:64|tau:0.05|D:512|H:8|L:6|predD:32|predH:2|predL:3|dropout:0.1|Wdecay:0.1|gradAcc:4|pUncond:0.1|cfg:2.5|useCnoise:False.pt' 
    jepa_edm_targetNet_path = './ckpts/jepa_edm_codeContests_targetNet|t0:0.002|tN:80|N_final:35|sampling:deterministic|lr:0.0003|batch:128|latDim:8|latSeq:64|tau:0.05|D:512|H:8|L:6|predD:32|predH:2|predL:3|dropout:0.1|Wdecay:0.1|gradAcc:4|pUncond:0.1|cfg:2.5|useCnoise:False.pt' 

    # set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

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
    train_dataset = train_dataset[:train_dataset_size]

    # init dit 
    dit_max_seq_len = t5_max_seq_len + 1 # [t, x]
    dit_condition_dim = jepa_latent_dim
    dit = init_dit(dit_max_seq_len, t5_max_seq_len, dit_d_model, dit_condition_dim, dit_vocab_size, dit_d_k, dit_d_v, dit_n_heads, dit_n_layers, dit_d_ff, dropout, device).to(device)

    # init optimizers
    dit_optimizer = torch.optim.AdamW(dit.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # load dit if resume 
    if resume_training_from_ckpt:
        dit, dit_optimizer = load_ckpt(dit_ckpt_path, dit, optimizer=dit_optimizer, device=device, mode='train')


    # init and load pretrained jepa models 
    jepa_max_seq_len_dit = t5_max_seq_len + 1 # [t, x_noised]
    jepa_condition_dim = t5_d_model 
    jepa_online_net = init_jepa_dit_discrete(jepa_max_seq_len_dit, t5_max_seq_len, jepa_latent_seq_len, jepa_d_model_dit, dit_vocab_size, jepa_latent_dim, jepa_condition_dim, jepa_d_k_dit, jepa_d_v_dit, jepa_n_heads_dit, jepa_n_layers_dit, jepa_d_ff_dit, dropout, device).to(device)

    jepa_pred_net = init_jepa_dit(jepa_max_seq_len_dit, t5_max_seq_len, jepa_latent_seq_len, jepa_d_model_pred_dit, jepa_latent_dim, jepa_latent_dim, jepa_condition_dim, jepa_d_k_pred_dit, jepa_d_v_pred_dit, jepa_n_heads_pred_dit, jepa_n_layers_pred_dit, jepa_d_ff_pred_dit, dropout, device).to(device)

    # jepa_target_net = deepcopy(jepa_online_net)

    jepa_online_net = load_ckpt(jepa_edm_onlineNet_path, jepa_online_net, device=device, mode='eval')
    jepa_pred_net = load_ckpt(jepa_edm_predNet_path, jepa_pred_net, device=device, mode='eval')
    # jepa_target_net = load_ckpt(jepa_edm_targetNet_path, jepa_target_net, device=device, mode='eval')


    # results for plotting
    results_dit_loss, results_dit_success_accuracy, results_dit_correct_accuracy = [], [], []
    results_sampleLoss_sedd, results_sampleLoss_jepaGen_sedd = [], []

    # train
    train_step = train_steps_done
    pbar = tqdm(total=num_train_steps)

    dit.train()

    # start DiT training
    while train_step < num_train_steps + train_steps_done:

        # fetch train minibatch
        idx = np.arange(len(train_dataset))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [train_dataset[i] for i in idx]

        # prepare program tokens
        with torch.no_grad():
            program_batch = [item['code'] for item in minibatch]
            program_tokens_dict = t5_tokenizer(program_batch, return_tensors='pt', padding='max_length', truncation=True)
            # TODO use obtained attention mask for fsq encoder 
            program_tokens, program_attn_mask = program_tokens_dict.input_ids, program_tokens_dict.attention_mask

            x = program_tokens.to(device) # x.shape: [b, seq_len] 

            # get caption embeddings
            cap_batch = [item['prompt'] for item in minibatch]
            cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
            cap_embs = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

            # get jepa encoding 
            if jepa_use_cnoise:
                t0 = torch.tensor(0.0001)
                t0 = 0.25 * torch.log(t0)
            else:
                t0 = torch.tensor(0.0) # TODO might want to use a less strict value eg ts[-2]
            t0 = t0.unsqueeze(-1).expand(x.shape[0]).to(device)
            online_latent = jepa_online_net(x, t0, cap_embs)
            pred_latent = prediction_function(jepa_pred_net, online_latent, t0, cap_embs, jepa_use_cnoise)

        ## loss for DiT 

        condition = pred_latent

        # set condition = None with prob p_uncond
        if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
            condition = None

        # sample diffusion time ~ uniform(eps, 1)
        t = (1 - diffusion_start_time_eps) * torch.rand(x.shape[0], device=device) + diffusion_start_time_eps

        # get noise from noise schedule
        sigma, dsigma = logLinearNoise(t)

        # perturb the data
        x_perturb = perturb(x, sigma, dit_mask_token)

        # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

        # get score
        log_score = dit(x_perturb, sigma, condition)

        # calculate loss 
        dit_loss = score_entropy_loss(log_score, sigma.unsqueeze(-1), x_perturb, x, dit_mask_token)
        dit_loss = (dsigma.unsqueeze(-1) * dit_loss).sum(dim=-1).mean()

        # adjustment for gradient accumulation 
        loss_scaled = dit_loss / gradient_accumulation_steps
        loss_scaled.backward()

        if (train_step + 1) % gradient_accumulation_steps == 0:
            # gradient cliping - helps to prevent unnecessary divergence 
            torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=1.0)
            # gradient step
            dit_optimizer.step()
            dit_optimizer.zero_grad()

        # bookeep losses
        results_dit_loss = ema(results_dit_loss, dit_loss.item()) 

        if len(results_dit_loss) > 0:
            pbar.set_description('dit_loss: {:.3f}'.format(results_dit_loss[-1]))

        pbar.update(1)


        # sampling
        if (train_step+1) % sampling_freq == 0:

            # put model in eval mode to avoid dropout
            dit.eval()

            pass

            dit.train()


        ## plotting
        if (train_step+0) % plot_freq == 0: ## save ckpt and plot losses

            # save dit ckpt 
            save_ckpt(device, dit_ckpt_path, dit, dit_optimizer)

            # plot dit loss
            if len(results_dit_loss) > 0:

                fig = plt.figure()
                plt.plot(results_dit_loss, label='dit_loss')
                plt.legend()
                plt.title('val:{:.3f}'.format(results_dit_loss[-1]))
                plt.ylim([0, 1000])
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '.png'
                fig.savefig(save_path)
                plt.close(fig)

            # # plot dit batch_accuracy
            # if len(results_dit_success_accuracy) > 0:

            #     fig = plt.figure()
            #     plt.plot(results_dit_success_accuracy, label='dit_success_accuracy')
            #     plt.plot(results_dit_correct_accuracy, label='dit_correct_accuracy')
            #     plt.legend()
            #     plt.title('success:{:.3f} correct:{:.3f}'.format(results_dit_success_accuracy[-1], results_dit_correct_accuracy[-1]))
            #     save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2) + '_ditAccuracy.png'
            #     fig.savefig(save_path)
            #     plt.close(fig)


        train_step += 1

    pbar.close()