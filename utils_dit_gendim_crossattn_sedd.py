### Utilities implementing Transformer backbone for a diffusion model
## Features:
# 1. From the original DiT Paper, the transformer backbone for diffusion model is not supposed to have a causal mask
# 2. This implementation just implements the non-causal decoder with input = [t, noised_x] and output = [denoised_x]. No separate 'final_emb' used.
# 3. For the condition_emb, we use an external encoder (e.g. t5_encoder for text modality). The condition_emb is fed to the decoder via xattn. 
# 4. Don't use sinusoidal embeddings for time since time here is diffusion time
# 5. Decoder layer is modified to skip xattn layer if condition_emb is None (for CFG)


import torch
import torch.nn as nn
from copy import deepcopy 
import torch.nn.functional as F
import numpy as np 
import math

## Transformer Modules 

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len] - this is the expected mask.shape for causal mask for nn.MultiheadAttention module
    return mask == 1  # True elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token) # mask.shape: [batch_size, max_seq_len] - this is the expected mask.shape for padding mask for nn.MultiheadAttention module
    return mask  # True elements are masked

# class implementing rotary positional embeddings 
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        self.d_model = d_model 
        self.device = device

        #init thetas
        i = torch.arange(0, d_model//2)
        thetas = torch.pow(max_seq_len, -(2*i)/d_model) # shape: [d_model/2]
        self.thetas = thetas.unsqueeze(-1).expand(-1, 2).flatten() # pairwise repeated entries, shape: [d_model] 
            

    def forward(self, x): # x.shape: [b, seqlen, d_model]
        batch_size, seqlen = x.shape[0], x.shape[1]

        # calculate rope
        indices = torch.arange(0, seqlen) # shape: [maxlen]
        frequencies = indices.unsqueeze(-1) * self.thetas.unsqueeze(0) # shape: [maxlen, d_model]
        cos_freq = torch.cos(frequencies).to(self.device)
        sin_freq = torch.sin(frequencies).to(self.device) 

        # apply rope
        x_cos = x * cos_freq.unsqueeze(0) 
        x = x.view(batch_size, seqlen, self.d_model//2, 2)
        x_odd, x_even = x[:, :, :, 0], x[:, :, :, 1] # x_odd.shape = [b, seqlen, d_model/2 ]
        x_shifted = torch.cat((-x_even.unsqueeze(-1), x_odd.unsqueeze(-1)), dim=-1) # x_shifted.shape: [b, seqlen, d_model/2, 2]
        x_shifted = x_shifted.flatten(start_dim=-2, end_dim=-1) # x_shifted.shape: [b, seqlen, d_model]
        x_sin = x_shifted * sin_freq.unsqueeze(0)
        x_rotated = x_cos + x_sin 
        return x_rotated

## Custom causal DiT Modules

# class to add positional encoding to embeddings (note that this class implements positional encoding as a constant untrainable vector)
class PositionalEncoding_Fixed(nn.Module):
    def __init__(self, d_model, maxlen):
        super().__init__()
        # calculate positional encoding and save them (register buffer for saving params in params_dict that are not to be updated during backprop)
        pe = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1) # pos.shape: [maxlen, 1]
        div_term = 10000.0 * torch.exp( torch.arange(0, d_model, 2) / d_model ) # div_term.shape: [d_model/2]
        pe[:, 0::2] = torch.sin(pos / div_term) # pe[:, 0::2].shape: [maxlen, d_model/2]
        pe[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pe", pe)
    # add positional encoding to the embedding - and freeze positional encoding
    def forward(self, x): # x.shape: [batch_size, seq_len, d_model]
        batch_size, seq_len = x.shape[0], x.shape[1]
        pos_emb = self.pe[:seq_len, :].requires_grad_(False) # [seq_len, d_model]
        pos_emb = pos_emb.expand(batch_size, -1, -1) # [b, seq_len, d_model]
        x = x + pos_emb 
        return x 
    
# class implementing residual + normalization connection - takes in any block and applies a normalization + residual connection
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return x + self.dropout(sublayer( self.norm(x) )) # note that we apply the norm first
    

# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.GELU()
    def forward(self, x):
        return self.w2(self.dropout( self.act_fn(self.w1(x)) ))

# class implementing multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, rope, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v 
        self.rope = rope 
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.output = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.attn_weights = None # placeholder to store attention weights (used to visualize attention matrices)
        self.dropout = dropout

    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, is_xattn=False, mask_padding=None, mask_causal=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        # key.shape: [batch_size, seq_len, d_model]; mask.shape: [batch_size, seq_len, seq_len]
        # project key, query, value and reshape into multiple heads
        batch_size = key.shape[0]
        causal_flag = not (mask_causal == None)

        # apply rotary embeddings - if self_attention
        if not is_xattn:
            # (batch_size, seq_len, d_model) -proj-> (batch_size, seq_len, proj_dim) -view-> (batch_size, seq_len, n_heads, d_k) -transpose-> (batch_size, n_heads, seq_len, d_k)
            proj_key = self.rope(self.W_K(key)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            proj_query = self.rope(self.W_Q(query)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        else:
            proj_key = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            proj_query = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # NOTE that rope is not applied to values
        proj_value = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # expand mask for n_heads
        if mask_padding is not None:
            mask_padding = mask_padding.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # mask.shape: [batch_size, n_heads, seq_len, seq_len]

        # calculate attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True): # context manager to enable flash attention
            # NOTE that torch's inbuilt scaled_dot_product_attention just needs the pad mask as input, it creates the causal mask automatically if is_causal is set to true
            attn_multihead = F.scaled_dot_product_attention(proj_query, proj_key, proj_value, mask_padding, self.dropout, is_causal=causal_flag)

        attn_multihead = attn_multihead.transpose(1, 2) # attn_multihead.shape: [batch_size, seq_len, n_heads, d_v]
        attn_multihead = torch.flatten(attn_multihead, start_dim=-2, end_dim=-1) # attn_multihead.shape: [batch_size, seq_len, n_heads * d_v]
        attn_multihead = self.output(attn_multihead) # attn_multihead.shape: [batch_size, seq_len, d_model]
        return attn_multihead

# NOTE transformer encoder not used 
# # class implementing a single encoder layer
# # each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
# class EncoderLayer(nn.Module):
#     def __init__(self, self_attn, feed_forward, dim, dropout):
#         super().__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
#     def forward(self, x, mask_padding, mask_causal):
#         x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask_padding=mask_padding, mask_causal=mask_causal)) # x.shape: [batch_size, seq_len, d_model]
#         x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
#         return x

# # class implementing the entire encoder block = stacked encoder layers
# class Encoder(nn.Module):
#     def __init__(self, layer, N, d_model):
#         super().__init__()
#         self.layers = clones(layer, N)
#         self.norm = nn.LayerNorm(d_model) # final layernorm at encoder output
#     def forward(self, x, mask_padding=None, mask_causal=None):
#         for layer in self.layers:
#             x = layer(x, mask_padding, mask_causal)
#         return self.norm(x)

# class implementing a single decoder layer
# each decoder layer has three blocks: 1. (self) (masked) multihead attention 2. (src) (unmasked) multihead x-attention  3. feed_forward; with sublayer connection around each
# NOTE Decoder layer is modified to skip xattn layer if condition_emb is None (for CFG)
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 3) # one for self_attn block, second for src_attn block, third for feed_forward block
    def forward(self, x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal):
        m = encoder_out
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, is_xattn=False, mask_padding=tgt_mask_padding, mask_causal=tgt_mask_causal)) # first apply self_attn block
        if m is not None:
            x = self.sublayers[1](x, lambda x: self.src_attn(m, x, m, is_xattn=True, mask_padding=src_mask_padding)) # src_attn: (key from encoder, query from decoder, value from encoder)
        x = self.sublayers[2](x, self.feed_forward)
        return x

# class implementing the entire decoder block = stacked decoder layers
class Decoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model) # final layernorm at decoder output
    def forward(self, x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal)
        return self.norm(x)


class DIT_Embedder(nn.Module):
    def __init__(self, max_seq_len_dit, x_seq_len, d_model, condition_dim, vocab_size, device):
        super().__init__()

        self.x_emb = nn.Embedding(vocab_size, d_model)
        torch.nn.init.kaiming_uniform_(self.x_emb.weight, a=math.sqrt(5))
        self.condition_emb = nn.Linear(condition_dim, d_model, bias=False)
        self.t_emb = nn.Linear(1, d_model, bias=False)
 
        self.d_model = d_model
        self.max_seq_len_dit = max_seq_len_dit # x_seq_len + 1 (for time)
        self.x_seq_len = x_seq_len
        self.device = device

    # function to get time embeddings from time int (based on sinusoidal position encoding)
    # NOTE fixed sinusoidal emb perform better than learnt linear emb for diffusion time (actually sigma) - opposite of what theory would suggest
    def get_time_embedding(self, t, d_model, maxval=100): # t.shape: [batch_size]
        t = t.unsqueeze(-1).float()
        inv_freq = 1.0 / (maxval ** (torch.arange(0, d_model, 2, device=self.device).float() / d_model))
        pos_enc_a = torch.sin(t.repeat(1, d_model // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, d_model // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    # function for forward prop 
    def forward(self,
                x,  # x.shape: [batch_size, x_seq_len]
                t,  # t.shape: [batch_size]
                condition_emb=None # shape: [batch_size, condition_seq_len, condition_dim]
                ):     
        batch_size = x.shape[0]
        x_emb = self.x_emb(x)

        if condition_emb is not None: # cfg
            condition_emb = self.condition_emb(condition_emb) 

        t_emb = self.get_time_embedding(t, self.d_model) # [b, d_model]
        t_emb = t_emb.unsqueeze(1) # [b, 1, d_model]

        # t_emb = self.t_emb(t.unsqueeze(-1))
        # t_emb = t_emb.unsqueeze(1) # [b, 1, d_model]

        dit_input_emb = torch.cat([t_emb, x_emb], dim=1) # dit_input_emb.shape: [batch_size, max_seq_len_dit, d_model]

        return dit_input_emb, condition_emb 


# class implementing the dit transformer (non-causal) constituting the diffusion backbone
class DIT(nn.Module):
    def __init__(self, embedder, decoder, max_seq_len_dit, x_seq_len, d_model, vocab_size, device):
        super().__init__()
        self.embedder = embedder
        self.decoder = decoder
        self.max_seq_len_dit = max_seq_len_dit # x_seq_len * 2 + 1 (for time)
        self.x_seq_len = x_seq_len 
        self.d_model = d_model

        self.final_proj = nn.Linear(d_model, vocab_size) # predict log_score
        self.final_proj.weight.data.zero_()
        self.final_proj.bias.data.zero_()

        self.device = device

    def forward(self, x, # x.shape: [b, x_seq_len] 
                    t, # diffusion time t.shape: [b]
                    condition=None, # shape: [b, condition_seq_len, condition_dim] # NOTE condition_emb is projected from condition_dim to d_model
                    seddType='absorb'
                ):
        dit_input_emb, condition_emb = self.embedder(x, t, condition) # dit_input_emb.shape: [batch_size, max_seq_len_dit, d_model]
        dit_out_seq = self.decoder(dit_input_emb, encoder_out=condition_emb, src_mask_padding=None, tgt_mask_padding=None, tgt_mask_causal=None) # dit_out_seq.shape: [batch_size, max_seq_len_dit, d_model]
        out = dit_out_seq[:, -self.x_seq_len:] # out.shape: [batch_size, x_seq_len, d_model]
        log_scores = self.final_proj(out) # log_score.shape: [batch_size, x_seq_len, vocab_size]

        if seddType == 'absorb':
            # scale and shift by sigma 
            esigm1_log = torch.where(t < 0.5, torch.expm1(t), t.exp() - 1).log().to(log_scores.dtype)[:, None, None]
            log_scores = log_scores - esigm1_log - np.log(log_scores.shape[-1] - 1)# this will be approximately averaged at 0

        # zero out the log scores for x = y along vocab dim
        log_scores = torch.scatter(log_scores, -1, x.unsqueeze(-1), torch.zeros_like(log_scores[..., :1]))

        return log_scores # the predicted log_score


# caller function to init dit
def init_dit(max_seq_len_dit, x_seq_len, d_model, condition_dim, vocab_size, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    assert max_seq_len_dit == x_seq_len + 1
    rope = RotaryPositionalEmbeddings(d_model, max_seq_len_dit ** 2, device)
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, rope, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    decoder_layer = DecoderLayer(deepcopy(attn), deepcopy(attn), deepcopy(ff), d_model, dropout) # single decoder layer
    decoder = Decoder(decoder_layer, n_layers, d_model) # decoder = stacked decoder layers
    # positional_encoder_fixed = PositionalEncoding_Fixed(d_model, max_seq_len_dit)
    dit_embedder = DIT_Embedder(max_seq_len_dit, x_seq_len, d_model, condition_dim, vocab_size, device)
    model = DIT(dit_embedder, decoder, max_seq_len_dit, x_seq_len, d_model, vocab_size, device)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# noise schedule
def logLinearNoise(t, eps = 1e-3):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)), i.e., the flip probability interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise sigma is -log(1 - (1 - eps) * t), so the sigma (derivative of total noise) will be (1 - eps) / (1 - (1 - eps) * t)
    """
    total_noise = -torch.log1p(-(1 - eps) * t)
    rate_noise = (1 - eps) / (1 - (1 - eps) * t)
    return total_noise, rate_noise 


# perturbation function for forward diffusion process (absorption / mask case)
def perturb(x, sigma, mask_token): # x.shape: [batch_size, seq_len]
    sigma = sigma.unsqueeze(-1) # sigma.shape: [batch_size, 1]
    flip_prob = 1 - (-sigma).exp()
    flip_indices = torch.rand(*x.shape, device=x.device) < flip_prob 
    x_perturb = torch.where(flip_indices, mask_token, x) # fill the mask_token at flip_indices; fill the original token at other indices
    return x_perturb # x_perturb.shape: [b, seqlen]


# loss function 
def score_entropy_loss(log_score, sigma, x, x0, mask_token): # log_score.shape: [b, seqlen, vocab_size]
    flipped_indices = x == mask_token # flipped_indices is a boolean tensor with shape: [b, seqlen]

    # calculate exp(sigma) - 1 with high precision
    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        torch.exp(sigma) - 1
    )

    # since ratio = p(y) / p(x) =
    # for unflipped indices = exp(-sigma) / exp(-sigma) = 1
    # for flipped indices = (1 - exp(-sigma)) / exp(-sigma) = exp(sigma) - 1 
    ratio = 1 / esigm1.expand_as(x)[flipped_indices] # ratio.shape: [b * num_flipped_tokens_in_each_sequence]
    flipped_tokens = x0[flipped_indices].unsqueeze(-1) # flipped_tokens.shape: [b * num_flipped_tokens_in_each_sequence, 1]

    ## prepare loss terms (equation 5 in the SEDD paper)

    # negative_term
    # torch.gather gathers the log_scores at the flipped indices (along seq_len_dim) and at the flipped token values (along the vocab dim) to give a 1-D tensor of shape [b * num_flipped_tokens_in_each_sequence]
    log_scores_for_flipped_indices = log_score[flipped_indices] # shape: [b * num_flipped_tokens_in_each_sequence, vocab_size]
    neg_term = ratio * torch.gather(log_scores_for_flipped_indices, -1, flipped_tokens).squeeze(-1)

    #positive term
    # sum all scores along the vocab dim, except for the mask_token
    pos_term = log_scores_for_flipped_indices[:, :-1].exp().sum(dim=-1) # shape: [b * num_flipped_tokens_in_each_sequence]

    # constant term
    const = ratio * (ratio.log() - 1)

    entropy = torch.zeros(*x.shape, device=x.device)
    entropy[flipped_indices] += pos_term - neg_term + const
    return entropy


# utility function to expand dims of x to match dims of y 
def unsqueeze_as(x, y):
    while len(x.shape) < len(y.shape):
        x = x.unsqueeze(-1)
    return x 

# utility function to sample from categorical distribution - TODO why not just use multinomial?
def sample_categorical(probs, sampling_eps):
    gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    sample = (probs / gumbel_norm).argmax(dim=-1)
    random_sample = torch.randint_like(sample, low=0, high=probs.shape[-1])
    eps = torch.rand(1) 
    if eps < sampling_eps:
        sample = random_sample
        sampling_eps = 0
    return sample, sampling_eps 

# utility function to calculate staggered score - corresponds to the LHS term in the product in equation 19 of the SEDD paper 
def get_stag_score(score, dsigma):
    '''
    score.shape: [b, seqlen, vocab_size]
    dsigma.shape: [b, 1]
    '''
    extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1) # extra_const.shape: [b=1, seqlen]
    stag_score = score * dsigma.exp().unsqueeze(-1)
    stag_score[..., -1] += extra_const # add extra_const to the score values for mask token transitions
    return stag_score

# utility function to calculate staggered probability = marginal probability but using dsigma = exp(dsigma * Q) - corresponds to the RHS term in the product in equation 19 of the SEDD paper 
def get_stag_prob(x, dsigma, vocab_size, mask_token):
    dsigma = unsqueeze_as(dsigma, x.unsqueeze(-1)) # dsigma.shape: [1, seqlen, 1]
    stag_prob = (-dsigma).exp() * F.one_hot(x, num_classes=vocab_size) # stag_prob.shape: [1, seqlen, vocab_size]
    stag_prob += torch.where(
        x == mask_token,
        1 - (-dsigma).squeeze(-1).exp(),
        0
    ).unsqueeze(-1)
    return stag_prob


## sampling function
def get_sample(net, seq_len, mask_token, vocab_size, num_sampling_steps, sample_batch_size, sample_condition, cfg_scale, sampling_eps, device, eps=1e-5):
    # x_T
    x = mask_token * torch.ones((sample_batch_size, seq_len), dtype=torch.int64, device=device)

    timesteps = torch.linspace(1, eps, num_sampling_steps + 1, device=device)
    dt = (1 - eps) / num_sampling_steps

    for i in range(num_sampling_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        curr_sigma = logLinearNoise(t)[0]
        next_sigma = logLinearNoise(t - dt)[0]
        dsigma = curr_sigma - next_sigma

        # get conditioned score
        log_score_cond = net(x, curr_sigma.squeeze(-1), sample_condition)
        score_cond = log_score_cond.exp().clone().detach()

        # get unconditioned score
        log_score = net(x, curr_sigma.squeeze(-1), None)
        score = log_score.exp().clone().detach()

        # apply cfg 
        score = cfg_scale * score_cond + (1 - cfg_scale) * score 

        # calculate staggered score - corresponds to the LHS term in the product in equation 19 of the SEDD paper 
        stag_score = get_stag_score(score, dsigma)

        # calculate staggered probability = marginal probability but using dsigma = exp(dsigma * Q) - corresponds to the RHS term in the product in equation 19 of the SEDD paper 
        stag_prob = get_stag_prob(x, dsigma, vocab_size, mask_token)

        # sampling probability for reverse diffusion process - equation 19 of SEDD paper
        probs = stag_score * stag_prob
        # x_(t-1)
        x, sampling_eps =  sample_categorical(probs, sampling_eps)
        
    ## final sampling step: going from x_1 to x_0

    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    sigma = logLinearNoise(t)[0]

    log_score_cond = net(x, sigma.squeeze(-1), sample_condition)
    score_cond = log_score_cond.exp().clone().detach()

    log_score = net(x, sigma.squeeze(-1), None)
    score = log_score.exp().clone().detach()

    # apply cfg
    score = cfg_scale * score_cond + (1 - cfg_scale) * score 

    stag_score = get_stag_score(score, sigma)

    stag_prob = get_stag_prob(x, sigma, vocab_size, mask_token)

    # added cfg 
    probs = stag_score * stag_prob

    # truncate probabilities - avoid mask prob
    probs = probs[..., :-1]
    x, sampling_eps =  sample_categorical(probs, sampling_eps)
        
    return x

