# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def MHA_wrapper(query, key, value, n_heads=1, causal=False):
    """
    This is a wrapper around the PyTorch implementation of multi-head attention.
    You will use this implementation to compare to your implementation for code testing.
    """
    assert query.shape == key.shape == value.shape
    _, n_tok, n_embd = query.shape

    query = query.transpose(0,1)
    key = key.transpose(0,1)
    value = value.transpose(0,1)

    in_proj_weight = torch.eye(n_embd, dtype=key.dtype, device=key.device).repeat((3, 1))
    out_proj_weight = torch.eye(n_embd, dtype=key.dtype, device=key.device)

    attn_mask = None
    if causal:
        attn_mask = torch.tril(torch.ones(n_tok, n_tok, dtype=bool, device=key.device)).logical_not()

    out, _ = F.multi_head_attention_forward(
        query, key, value, n_embd, n_heads,
        in_proj_weight=in_proj_weight, in_proj_bias=None,
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
        out_proj_weight=out_proj_weight, out_proj_bias=None,
        attn_mask=attn_mask, need_weights=False,)

    return out.transpose(0,1)


import torch # Add torch import here

# use cpu for now
DEVICE = 'cpu'

# make these bigger if you want a stricter test of your code
part1_n_tok = 10
part1_n_emb = 6

# generate fixed pseudo-random Q,K,V for testing attn function
torch.manual_seed(447)

# Initialize random testing Q,K,V
part1_key = torch.randn(1, part1_n_tok, part1_n_emb)
part1_value = torch.randn(1, part1_n_tok, part1_n_emb)
part1_query = torch.randn(1, part1_n_tok, part1_n_emb)


def init_qkv_proj(n_embd:int):
    """
    return: A tuple of length 3 containing the projections for Q, K, V.
    """
    return (nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd))

def self_attention(Q, K, V, n_heads=1, causal=True):

    assert Q.shape == K.shape == V.shape
    B, n_tok, n_embd = Q.size()

    # TODO: Step 3 -- split heads. Only do this after you finish coding STEP 3 and before you Test it
    if n_heads > 1:
      d_head = n_embd // n_heads
      Q = Q.view(B, n_tok, n_heads, d_head).transpose(1, 2)
      K = K.view(B, n_tok, n_heads, d_head).transpose(1, 2)
      V = V.view(B, n_tok, n_heads, d_head).transpose(1, 2)

    A = torch.matmul(Q, K.transpose(-1, -2))
    A = A / (Q.size(-1) ** 0.5)

    if causal:
      n_tok = A.size(-1)
      mask = torch.tril(torch.ones(n_tok, n_tok, dtype=bool, device=A.device))

      while mask.dim () < A.dim ():
        mask = mask.unsqueeze(0)

      A = A.masked_fill(~mask, float('-inf'))

    A = torch.softmax(A, dim=-1)
    y = torch.matmul(A, V)

    if n_heads > 1:
      y = y.transpose(1, 2).contiguous().view(B, n_tok, n_embd)


    assert y.shape == (B, n_tok, n_embd)
    return y


def pairwise_similarities(Q, K):

    A = torch.matmul(Q, K.transpose(-1, -2))
    return A

def attn_scaled(A, n_embd:float, n_heads:float):

    assert n_embd % n_heads == 0,
    # TODO:
    d_head = n_embd // n_heads
    A = A / (d_head ** 0.5)
    return A

def attn_softmax(A):
    A = torch.softmax(A, dim=-1)

    return A

def compute_outputs(A, V):
  
    out = torch.matmul(A, V)
    return out


out_A = self_attention(part1_query, part1_key, part1_value, n_heads=1, causal=False)
out_B = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=1, causal=False)
assert out_A.shape == out_B.shape == part1_query.shape

print('max diff:', (out_A - out_B).abs().max().item())


def make_causal_mask(n_tok:int):
  
    mask = torch.tril(torch.ones(n_tok, n_tok, dtype=torch.bool))
    return mask

def apply_causal_mask(mask, A):
   
    while mask.dim() < A.dim():
      mask = mask.unsqueeze(0)

    A_masked = A.masked_fill(~mask, float('-inf'))
    return A_masked


out_A = self_attention(part1_query, part1_key, part1_value, n_heads=1, causal=True)
out_B = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=1, causal=True)
assert out_A.shape == out_B.shape == part1_query.shape

print('max diff:', (out_A - out_B).abs().max().item())

def split_heads_qkv(Q, K, V, n_heads:int):

    return (split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads))

def split_heads(x, n_heads:int):

    B, n_tok, n_embd = x.size()
    assert n_embd % n_heads == 0, "d must be divisible by number of heads"
    d_head = n_embd // n_heads
    x = x.view(B, n_tok, n_heads, d_head).transpose(1, 2)
    return x


def merge_heads(y):

    B, nh, n_tok, nc = y.size()
    y = y.transpose(1, 2).contiguous().view(B, n_tok, nh * nc)
    return y



out_A = self_attention(part1_query, part1_key, part1_value, n_heads=3, causal=True)
out_B = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=3, causal=True)
assert out_A.shape == out_B.shape == part1_query.shape

print('max diff:', (out_A - out_B).abs().max().item())


import torch

def self_attention_mask_after_softmax(Q, K, V, n_heads=1, causal=True):
    B, T, D = Q.shape

    # split into heads (same pattern as §1.1)
    if n_heads > 1:
        d = D // n_heads
        Q = Q.view(B, T, n_heads, d).transpose(1, 2)
        K = K.view(B, T, n_heads, d).transpose(1, 2)
        V = V.view(B, T, n_heads, d).transpose(1, 2)
    else:
        d = D

    # raw scores + scaling
    A = Q @ K.transpose(-1, -2)
    A = A / (d ** 0.5)

    # *** key ablation: softmax FIRST ***
    A = torch.softmax(A, dim=-1)

    if causal:
        # lower-triangular mask, broadcast to A's dims
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=A.device))
        while mask.dim() < A.dim():
            mask = mask.unsqueeze(0)

        # zero illegal future probs AFTER softmax (NO renorm)
        A = A * mask
        #ACTIVE STATUS
        if not hasattr(self_attention_mask_after_softmax, "_ping"):
            print("[§1.2] post-softmax mask active (no renorm). "
                  f"mean rowsum after mask: {A.sum(-1).mean().item():.4f}")
            self_attention_mask_after_softmax._ping = True

    # weighted sum
    Y = A @ V

    # merge heads back
    if n_heads > 1:
        Y = Y.transpose(1, 2).contiguous().view(B, T, D)

    return Y



### Utilities, data, and imports


![ -e "N-gram.zip" ] || gdown 1MtgMIE1ghyw4pShjSLIb0b2xKkzi7nsk
!unzip -o N-gram.zip

# clone our fork of minGPT and link to the code
![ -d "mingpt-cse447" ] || git clone https://gitlab.cs.washington.edu/yegork/mingpt-cse447.git
![ -e "mingpt" ] || ln -s mingpt-cse447/mingpt mingpt

from mingpt.model import GPT
from mingpt.trainer import Trainer

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
import numpy as np

"""### Dataset processing"""


!head 1b_benchmark.train.tokens

with open('1b_benchmark.train.tokens', 'r') as f: lines_train = f.readlines()
with open('1b_benchmark.dev.tokens', 'r') as f: lines_dev = f.readlines()
with open('1b_benchmark.test.tokens', 'r') as f: lines_test = f.readlines()

tokens_train = [line.split() for line in lines_train]

print(f'train docs: {len(tokens_train)}')
print(f'total train tokens: {sum(len(t) for t in tokens_train)}')

def flat(tokens):
    for t in tokens:
        yield from t


token_counts = Counter(flat(tokens_train))
token_counts['<START>'] = 1000004
token_counts['<STOP>'] = 1000003
token_counts['<UNK>'] = 1000002
token_counts['<PAD>'] = 1000001
sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

print('unique_tokens:', len(token_counts))
print('unique_tokens, count>=3:', len([t for t in sorted_tokens if t[1] >= 3]))


tokenizer = {t[0]: i for i, t in enumerate(sorted_tokens) if t[1] >= 3}

def pad_to_length(tokens, max_len, tokenizer=tokenizer):
    return tokens[:max_len] + [tokenizer['<PAD>']] * (max_len - len(tokens))

def tokenize(sentence, pad_to_len=None, include_stop=True, tokenizer=tokenizer):
    words = [tokenizer.get(w, tokenizer['<UNK>']) for w in sentence.split()]
    tokens = [tokenizer['<START>']] + words + ([tokenizer['<STOP>']] * include_stop)

    if pad_to_len is not None:
        tokens = pad_to_length(tokens, pad_to_len, tokenizer=tokenizer)
    return tokens

tokenizer_inv = {v:k for k,v in tokenizer.items()}
def decode(tokens, tokenizer_inv=tokenizer_inv, end_at_stop=True, omit_pad=True):
    tokens = [tokenizer_inv[t] for t in tokens]
    if omit_pad:
        tokens = [t for t in tokens if t != '<PAD>']
    if end_at_stop and '<STOP>' in tokens:
        tokens = tokens[:tokens.index('<STOP>')+1]
    return ' '.join(tokens)


sentence = 'More people have said an Escher sentence than I have .'
tokenized = tokenize(sentence, pad_to_len=25) # pad to only 25 so it looks nice
decoded = decode(tokenized, end_at_stop=False, omit_pad=False)
print(f'{sentence=}\n{tokenized=}\n{decoded=}')

plt.hist([len(t) for t in tokens_train], bins=50)
plt.title('sequence lengths in the train dataset by tokens')
plt.ylabel('# of sequences')
plt.xlabel('sequence length in tokens')
plt.show()

# Notice above that the vast majority of sequences have less than 100 tokens.
# For performance we will thus truncate to 100 tokens.

MAX_LEN = 100
DEVICE = 'cuda'

data_train = torch.tensor(
    [tokenize(t, MAX_LEN) for t in lines_train if len(t) > 0],
    dtype=torch.long
)
data_val = torch.tensor(
    [tokenize(t, MAX_LEN) for t in lines_dev if len(t) > 0],
    dtype=torch.long
)

data_train.shape, data_val.shape

# X is all but last token, Y is all but first token
train_dataset = torch.utils.data.TensorDataset(data_train[:, :-1], data_train[:, 1:])
val_dataset = torch.utils.data.TensorDataset(data_val[:, :-1], data_val[:, 1:])

# example X,Y pair from train dataset -- 2 is <START>, 3 is <STOP>
train_dataset[447]

"""
### Model and Trainer code"""

model_config = GPT.get_default_config()
model_config.model_type = None
model_config.pad_token = tokenizer['<PAD>']


model_config.model_type = 'gpt-nano'
# 'gpt-nano' equivalent to:
# model_config.n_layer = 3
# model_config.n_head = 3
# model_config.n_embd = 48

model_config.vocab_size = max(tokenizer.values()) + 1
# model_config.vocab_size = 50257 # openai's model vocabulary, if using gpt2 BPE


model_config.block_size = 1024

model_config.attn_init_fn = init_qkv_proj
model_config.attn_fn = self_attention_mask_after_softmax

# Can use the wrapper around PyTorch's multi-head attention instead, but it's hard to modify for experiments
# model_config.attn_fn = MHA_wrapper

model = GPT(model_config)

import numpy as np
np.save('variant_log.npy', np.array(log, dtype=float))
print("Saved variant_log.npy")

train_config = Trainer.get_default_config()
train_config.device = DEVICE
train_config.num_workers = 2

train_config.learning_rate = 5e-4
train_config.batch_size = 32
train_config.max_iters = len(train_dataset) // train_config.batch_size  # train for 1 epoch

trainer = Trainer(train_config, model, train_dataset)
log = []

model.to(DEVICE)
model.train()

bar = tqdm(total=train_config.max_iters)
@torch.no_grad()
def on_batch_end(trainer):
    log.append( trainer.loss.item() )
    bar.set_postfix(loss=trainer.loss.item())
    bar.update()

trainer.set_callback('on_batch_end', on_batch_end)
trainer.run()
bar.close()

plt.plot(log)
plt.xlabel('step')
plt.ylabel('loss')
plt.show()



"""### Evaluation
"""

sentence = 'Thank you so much Liwei and Taylor for all your help with this !'

tokens = torch.tensor([tokenize(sentence, pad_to_len=MAX_LEN)], dtype=torch.long)
X_tokens, y_tokens = tokens[:, :-1], tokens[:, 1:]

print('notice the long tail of PAD tokens: ', tokens.cpu()[0].tolist())

model.eval()
with torch.no_grad():
    logits, loss = model(X_tokens.to(DEVICE), y_tokens.to(DEVICE))
    logits, loss = logits.cpu(), loss.cpu()


also_loss = F.cross_entropy(logits.flatten(0,1), y_tokens.flatten(0,1),
                            ignore_index=tokenizer['<PAD>'])


probs = F.softmax(logits, dim=-1)

# work with log of the probabilities for numerical stability
log_probs = torch.log(probs)

# this is weird pytorch screwery to index into last dimension of log_probs with y_tokens
# this selects only the log probabilities of the target tokens
y_log_probs = torch.gather(log_probs, -1, y_tokens[..., None])[..., 0]

# get all the target log probabilities EXCEPT for when that target token is <PAD>
not_pad_y_log_probs = y_log_probs[y_tokens != tokenizer['<PAD>']]

# negative average of the log probs of the target tokens is exactly crossentropy loss here!
also_loss_again = -not_pad_y_log_probs.mean()

print()
print('reported loss from model:\t', loss.item())
print('manually calculated loss:\t', also_loss.item())
print('manually calculated loss again:\t', also_loss_again.item())

# we can calculate perplexity using the crossentropy loss
perplexity = torch.exp(also_loss)
print('perplexity:', perplexity.item())

"""
We've made a utility function to calculate loss per-document for some data.
It accepts a list of strings, tokenizes, evaluates, and returns a list of floats.
"""
@torch.no_grad
def evaluate_losses(data, model=model, bs=32, progress=True, pad_to_len=MAX_LEN):
    it = range(0, len(data), bs)
    if progress: it = tqdm(it)

    out = []
    for b_start in it:
        batch = slice(b_start, b_start+bs)
        tokens = torch.tensor(
            [tokenize(t, pad_to_len=pad_to_len) for t in data[batch]],
            dtype=torch.long).to(DEVICE)
        X_tokens, y_tokens = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        model.eval()
        logits, _ = model(X_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]

        for i in range(y_tokens.shape[0]):
            not_pad = (y_tokens[i] != tokenizer['<PAD>'])
            loss = -y_log_probs[i, not_pad].mean()
            out.append(loss.item())

    return out

# calculate loss and perplexity for a single sentence
is_this_loss = evaluate_losses(['After learning language models model natural language',], progress=False)[0]
print('loss:', is_this_loss)
print('perplexity:', np.exp(is_this_loss))

train_losses = evaluate_losses(lines_train)
print('train perplexity:', np.mean(np.exp(train_losses)))

dev_losses = evaluate_losses(lines_dev)
print('dev perplexity:', np.mean(np.exp(dev_losses)))

import numpy as np

baseline_ppl = float(np.mean(np.exp(dev_losses)))   # <-- make the number
np.save('baseline_log.npy', np.array(log, dtype=float))  # training curve

with open('ppl.txt','w') as f:                      # overwrite for baseline
    f.write(f"baseline {baseline_ppl}\n")

print("Saved baseline.")




sentence = ''                         # empty prompt -> sample from model at random
# sentence = 'unfortunately ,'          # can sample more negative stuff
# sentence = 'fun fact : did you know'  # AI-generated fun facts

tokens = torch.tensor([tokenize(sentence, include_stop=False)], dtype=torch.long).to(DEVICE)

for _ in range(10):
    pred = model.generate(tokens, MAX_LEN-tokens.shape[-1],
                        temperature=1.0, do_sample=True, top_k=None)

    print(decode(pred[0].tolist()))
