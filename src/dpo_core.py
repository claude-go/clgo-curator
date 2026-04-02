"""Coeur DPO — calcul de logprobs et loss.

Le DPOTrainer de mlx-tune a un bug : sans ref_model, il utilise
stop_gradient sur le meme forward pass, ce qui annule le signal
de gradient (loss = log(2) constant).

Ce module pre-calcule les logprobs de reference avant le training.
"""

import mlx.core as mx
import mlx.nn as nn


def compute_logprobs(model, tokenizer, text, max_len=512):
    """Calcule les log-probabilites moyennes par token d'un texte."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    if len(tokens) < 2:
        return mx.array(0.0)

    x = mx.array(tokens[:-1])[None, :]
    targets = mx.array(tokens[1:])

    logits = model(x)[0]
    log_probs = nn.log_softmax(logits, axis=-1)
    token_log_probs = log_probs[mx.arange(len(targets)), targets]

    return mx.mean(token_log_probs)


def precompute_reference(model, tokenizer, dataset, max_len=512):
    """Pre-calcule les logprobs de reference (frozen)."""
    ref_chosen = []
    ref_rejected = []

    for i, pair in enumerate(dataset):
        chosen_text = pair["prompt"] + " " + pair["chosen"]
        rejected_text = pair["prompt"] + " " + pair["rejected"]

        lp_chosen = compute_logprobs(model, tokenizer, chosen_text, max_len)
        lp_rejected = compute_logprobs(
            model, tokenizer, rejected_text, max_len,
        )
        mx.eval(lp_chosen, lp_rejected)

        ref_chosen.append(lp_chosen.item())
        ref_rejected.append(lp_rejected.item())

        if (i + 1) % 50 == 0:
            print(f"  reference: {i + 1}/{len(dataset)}")

    return ref_chosen, ref_rejected


def dpo_loss_fn(model, tokenizer, pair, ref_c, ref_r, beta, max_len):
    """Calcule la loss DPO pour une paire."""
    chosen_text = pair["prompt"] + " " + pair["chosen"]
    rejected_text = pair["prompt"] + " " + pair["rejected"]

    pi_chosen = compute_logprobs(model, tokenizer, chosen_text, max_len)
    pi_rejected = compute_logprobs(
        model, tokenizer, rejected_text, max_len,
    )

    log_ratio_c = pi_chosen - ref_c
    log_ratio_r = pi_rejected - ref_r

    logits = beta * (log_ratio_c - log_ratio_r)
    loss = -nn.log_sigmoid(logits)

    return loss
