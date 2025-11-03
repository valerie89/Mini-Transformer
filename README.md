# Mini Transformer — Attention from Scratch
Overview

This project implements self-attention and multi-head attention from scratch using PyTorch.
It follows the foundation of transformer architectures such as GPT and was completed for the BC3705 Natural Language Processing course.

The project has two parts:

Part 1.1: Implementing attention from scratch (single-head, causal, and multi-head)

Part 1.2: Experimenting with a transformer variant (changing the order of masking and normalization)

Features

Manual implementation of attention and multi-head attention

Causal masking for autoregressive modeling

Custom variant of self-attention for ablation testing

Integration with a small GPT model (minGPT-nano)

Token-level perplexity evaluation

Implementation Details
Attention Core Functions

pairwise_similarities(Q, K) — computes raw attention scores

attn_scaled(A, n_embd, n_heads) — scales scores by √d

attn_softmax(A) — normalizes attention weights

compute_outputs(A, V) — computes weighted sums of values

make_causal_mask(n_tok) — generates a lower-triangular causal mask

apply_causal_mask(mask, A) — applies mask to block future tokens

Multi-Head Extension

split_heads() — splits embedding dimensions across heads

merge_heads() — merges outputs back into full embedding size

Experimental Variant

A modified version of self-attention (self_attention_mask_after_softmax) was created to test how the order of masking affects learning.

Implementation	Masking Order
Baseline	Mask → Softmax
Variant	Softmax → Mask → Renormalize
Training Setup

Dataset: One Billion Word Benchmark (University of Washington subset)

Tokenizer: Word-level with <START>, <STOP>, <PAD>, <UNK> tokens

Model: GPT-nano (3 layers, 3 heads, 48 embedding dimensions)

Framework: PyTorch + minGPT

Hardware: Google Colab GPU (T4)

Loss: Cross-entropy (ignoring <PAD> tokens)

Metric: Per-document perplexity

Results
Model	Train PPL	Dev PPL	Notes
Baseline (mask → softmax)	402.97	383.19	Standard causal attention
Variant (softmax → mask + renorm)	182.98	170.78	Lower perplexity, noisier samples

Key Observations

The variant converged faster and reached lower perplexity.

Text quality decreased slightly (more <UNK> tokens).

Both models remained causal but differed in gradient dynamics.

How to Run

Open the notebook in Google Colab.

Go to Runtime → Change runtime type → GPU (T4).

Run all cells in order.

To switch between implementations, edit:

# Baseline
model_config.attn_fn = self_attention

# Variant
model_config.attn_fn = self_attention_mask_after_softmax


Re-run training and evaluation sections.
