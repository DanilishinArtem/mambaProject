import numpy as np
import torch
import string
import torch.nn.functional as F
import random 
import matplotlib.pyplot as plt
import numpy as np

class NumberTokenizer:
    def __init__(self, TO_TOKEN, TO_CHAR):
        
        self.TO_TOKEN = TO_TOKEN
        self.TO_CHAR = TO_CHAR

        self.bos_token_id = TO_TOKEN['$']
        self.eos_token_id = TO_TOKEN['.']

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def decode(self, x):
        x = x.detach().cpu().numpy()
        decoded = ''.join([str(t) if t not in self.TO_CHAR else self.TO_CHAR[t] for t in x])
        return decoded

    def __len__(self):
        return len(self.TO_TOKEN)


def arr_to_str(x):
    return ''.join([str(n) for n in x])

def rand_num(length,vocab_size):
        string_ascii_lowercase = string.ascii_lowercase[:vocab_size]
        num = "".join(np.random.choice(list(string_ascii_lowercase),size=length))
        return arr_to_str(num)


def generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size):

    counter_max_ngram = 10
    counter_ngram = 0
    unique = False
    while not unique:
       num1 = rand_num(len1,vocab_size)
       max_limit = len(num1) - n_gram - length_answer -1 if length_answer > 0 else len(num1) - n_gram -1
       list_ngrams = [num1[idx : idx + n_gram] for idx in range(max_limit)]
       unique_n_grams = []
       for ng in list_ngrams:
         if list_ngrams.count(ng) == 1:
           unique_n_grams.append(ng)
       if unique_n_grams:
         unique = True
       counter_ngram +=1
       if counter_ngram >= counter_max_ngram:
          raise ValueError(f"Unable to find a unique {n_gram}-gram in a string of length {len1}!")
    return num1, list_ngrams  


def sample_str(len1,n_gram,length_answer,task,vocab_size=26):

    if task=="copy":
       num1 = rand_num(len1,vocab_size)
       answer = num1[:length_answer] if length_answer > 0 else num1
       example_str = f'${num1}|{answer}.'
    
    elif task == "duplicate_ngram":
       ### sample strings until having one that has a unique n-gram
       num1, list_ngrams = generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size) 
       ngram_new, ngram_old =  random.sample(list_ngrams, 2)

       #create a string with duplicate ngrams
       num1 = num1.replace(ngram_old,ngram_new)

       example_str = f'${num1}|{num1}.'
    elif task in ["prefix_ngram","suffix_ngram"]:
       
       ### sample strings until having one that has a unique n-gram
       num1, list_ngrams = generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size) 

       ngram = list_ngrams[np.random.randint(low=0,high=len(list_ngrams)-1,size=1)[0]]
       index_ngram = num1.index(ngram)

       next_chunk = num1[(index_ngram + len(ngram)):]
       answer = next_chunk[:(length_answer)] if length_answer > 0 else next_chunk

       if task == "prefix_ngram":
          example_str = f'${ngram}|{num1}|{answer}.'
       elif task == "suffix_ngram":
          example_str = f'${num1}|{ngram}{answer}.'
    return example_str 

class CopyDataset:
    def __init__(self, tokenizer, vocab_size=26, n_gram=3, length_answer=-1, train_task="copy", sequence_length=220, min_length=20, max_length=50, num_examples=1000, batch_size=8): 
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.train_task = train_task
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        self.length_answer = length_answer

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'mask': []}
        
        minimal_required_length = self.n_gram if self.n_gram > 0 else 0
        minimal_required_length += self.length_answer if self.length_answer > 0 else 0
        if self.min_length <= minimal_required_length:
            raise ValueError(f"Minimum length is set to {self.min_length} and is smaller than the required one {minimal_required_length}")
        
        minimal_required_length = self.max_length*2
        if self.sequence_length <= minimal_required_length:
            raise ValueError(f"Strings of size {self.max_length} do not fit in a context of size {self.sequence_length} because {2*self.max_length}>{self.sequence_length}. Increase your context length !")

        for _ in range(self.batch_size):
            prospective_len = 0
            full_str = ""
            example_mask = []
            while prospective_len < self.sequence_length:
              
              ##sample a string  
              len1 = np.random.randint(self.min_length, self.max_length+1)
              example_str = sample_str(len1,self.n_gram,self.length_answer,self.train_task,self.vocab_size)
              
              ###setting up mask for training loss 
              if self.train_task=="copy":
                 example_mask_tmp = [0] * (len1+2) + [1] * (len(example_str) - len1-2)
              elif self.train_task=="prefix_ngram":
                 example_mask_tmp = [0] * (len1+(self.n_gram+3)) + [1] * (len(example_str) - len1-(self.n_gram+3))
              elif self.train_task == "suffix_ngram":
                 example_mask_tmp = [0] * (len1+self.n_gram+2) + [1] * (len(example_str) - (len1+self.n_gram+2))
              

              #packing the context with examples
              if prospective_len+len(example_str) > self.sequence_length:
                 remaining_len = self.sequence_length - prospective_len
                 remaining_mask_len = self.sequence_length - prospective_len
                 full_str += example_str[:remaining_len]
                 example_mask += [0]*(remaining_mask_len)
                 break
              else:
                 full_str += example_str
                 prospective_len += len(example_str)
                 example_mask += example_mask_tmp

            assert len(full_str) == len(example_mask)
            example_ids = self.tokenizer(full_str)
            example_mask = torch.tensor(example_mask)

            batch['input'].append(full_str)
            batch['input_ids'].append(example_ids)
            batch['mask'].append(example_mask)
        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch

def get_tokenizer(vocab_size):
    string_ascii_lowercase = string.ascii_lowercase[:vocab_size]
    letters = dict(zip(string_ascii_lowercase, range(vocab_size)))

    symbols = {'$': len(letters), '|': len(letters)+1, '.': len(letters)+2, '*': len(letters)+3}

    TO_TOKEN = {**letters, **symbols}

    TO_CHAR = {v:k for k,v in TO_TOKEN.items()}

    tokenizer = NumberTokenizer(TO_TOKEN, TO_CHAR)
    return tokenizer, TO_TOKEN, TO_CHAR

def batch_covariance(x: torch.Tensor, by_time: bool = True) -> torch.Tensor:
    """
    Вычисляет ковариацию.
    Если by_time=True → ковариация между временными шагами.
    Если False → ковариация между признаками.
    """
    if by_time:
        # Считаем ковариацию по оси seq_len
        # x: [batch, seq_len, dim] → [batch, dim, seq_len]
        x = x.transpose(1, 2)
    # Центрируем
    x_centered = x - x.mean(dim=-1, keepdim=True)
    cov = torch.matmul(x_centered, x_centered.transpose(-1, -2))
    cov = cov / (x.shape[-1] - 1)
    return cov


def attn_score_plot(attn_scores, path: str, name: str):
    print("[INFO] start of working error_hist_plot for {}".format(name))
    slice_ = False
    t_mark = 100
    attn_score_heatmap = attn_scores.mean(dim=0)
    attn_score_heatmap[attn_score_heatmap == float('-inf')] = -3
    attn_score_heatmap_min = attn_score_heatmap.min(dim=0, keepdim=True)[0]
    attn_score_heatmap = attn_score_heatmap - attn_score_heatmap_min
    mask = torch.triu(torch.ones_like(attn_score_heatmap), diagonal=1).bool()
    attn_score_heatmap[mask] = float('nan')

    mean_attn = attn_scores.mean(dim=0)
    if slice_:
        time = torch.arange(mean_attn.shape[0])
        t_grid, s_grid = torch.meshgrid(time, time, indexing='ij')
        grid = (t_grid - s_grid)[:,t_mark]
        mean_attn = mean_attn[:,t_mark]
        relative_pos = grid.flatten()
        attn_values = mean_attn.flatten()
    else:
        time = torch.arange(mean_attn.shape[0])
        t_grid, s_grid = torch.meshgrid(time, time, indexing='ij')
        relative_pos = (t_grid - s_grid).flatten()
        attn_values = mean_attn.flatten()

    attn_values[attn_values == -float('inf')] = float('inf')
    attn_values[attn_values < -10000] = float('inf')
    attn_values_min = attn_values.min(dim=0, keepdim=True)[0]
    attn_values = attn_values - attn_values_min
    # Group by each unique relative position
    from collections import defaultdict
    bins = defaultdict(list)

    for delta, val in zip(relative_pos.tolist(), attn_values.tolist()):
        if delta > 0:
            bins[delta].append(val)

    relative_pos_vals = sorted(bins.keys())
    mean_values = [sum(bins[k])/len(bins[k]) for k in relative_pos_vals]

    # Отрисовка ковариационной функции ...
    plt.figure(figsize=(10, 4))
    plt.plot(relative_pos_vals, mean_values, marker='o')
    plt.xlabel("t - s (relative position)")
    plt.ylabel("Average Attention")
    plt.title("u_s = f(t - s)")
    plt.grid(True)
    plt.savefig("{}/covar_{}.png".format(path, name))

    # Отрисовка хитмапы ...
    attn_score_heatmap = attn_score_heatmap.numpy()
    plt.figure(figsize=(10, 5))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    plt.imshow(attn_score_heatmap, cmap=cmap, interpolation='none')
    plt.colorbar(label='Value')
    plt.xlabel('m')
    plt.ylabel('n')
    plt.title('Heatmap')
    plt.savefig("{}/heatmap_{}.png".format(path, name))

import torch.nn as nn
class BayesianPrefixFilter(nn.Module):
    def __init__(self, dim, init_log_obs_var=0.0, init_log_proc_var=-6.0, init_decay_logit=0.0, eps=1e-6):
        super().__init__()
        self.dim = dim
        # learnable observation variance (per-dim)
        self.log_obs_var = nn.Parameter(torch.full((dim,), float(init_log_obs_var)))
        # learnable process (dynamics) variance (per-dim)
        self.log_proc_var = nn.Parameter(torch.full((dim,), float(init_log_proc_var)))
        # learnable decay/logit that maps to (0,1) via sigmoid
        self.decay_logit = nn.Parameter(torch.tensor(float(init_decay_logit)))
        self.eps = eps

    def forward(self, X, init_mean=None, init_var=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype
        assert D == self.dim, (D, self.dim)

        # per-dim variances (positive) broadcastable to (B, D)
        obs_var = F.softplus(self.log_obs_var).view(1, D).to(device=device, dtype=dtype)  # (1, D)
        proc_var = F.softplus(self.log_proc_var).view(1, D).to(device=device, dtype=dtype)  # (1, D)
        decay = torch.sigmoid(self.decay_logit).to(device=device, dtype=dtype)  # scalar in (0,1)

        # init prior
        if init_mean is None:
            prior_mean = torch.zeros(B, D, device=device, dtype=dtype)
        else:
            prior_mean = init_mean.to(device=device, dtype=dtype)
        if init_var is None:
            prior_var = torch.ones(B, D, device=device, dtype=dtype)  # moderate uncertainty to start
        else:
            prior_var = init_var.to(device=device, dtype=dtype)

        outputs = []
        # loop over time (recursive Bayesian update)
        for t in range(L):
            obs = X[:, t, :]  # (B, D)

            # Predict / process step: add process noise and optionally apply decay to mean
            prior_mean = prior_mean * decay                          # allow forgetting; decay in (0,1)
            prior_var = prior_var + proc_var                         # increase uncertainty

            # Bayes update (Gaussian conjugate)
            denom = prior_var + obs_var + self.eps                   # (B, D)
            post_var = (prior_var * obs_var) / denom                 # (B, D)
            post_mean = (obs_var * prior_mean + prior_var * obs) / denom

            outputs.append(post_mean)

            # set posterior as next prior
            prior_mean = post_mean
            prior_var = post_var

        post_means = torch.stack(outputs, dim=1)  # (B, L, D)
        return post_means
