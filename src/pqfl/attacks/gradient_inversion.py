"""Gradient inversion of a LoRA update (optimization-based, torch): the strongest-attacker positive control.

The attacker knows the model and the adapter, the prompt tokens and the completion length, and sees the
gradient of the example's mean completion NLL w.r.t. the trainable (LoRA) parameters. It optimizes soft
token distributions for the completion positions so that the gradient they induce matches the observed one
(cosine distance), through double back-propagation, then reads the tokens with argmax. Success is judged
with pqfl.attacks.inversion against a random-text null. Honest caveat: the LoRA gradient is a rank-8
projection of the activations, so recovery is not guaranteed even for this attacker.
"""
import torch
import torch.nn.functional as F

from pqfl.train.local import trainable


_KERNEL_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update")


def use_reference_kernels() -> dict:
    """Replace the fused (fla / flash) implementations of the Qwen3.5 linear-attention layers by their pure
    PyTorch originals, which support double backward. Returns the replaced functions: pass it to
    restore_kernels() when done, otherwise every later job of the same process trains with the slow
    reference kernels."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    saved = {}
    for name in _KERNEL_NAMES:
        function = getattr(modeling, name, None)
        original = getattr(function, "__wrapped__", None)
        if original is not None:
            saved[name] = function
            setattr(modeling, name, original)
    return saved


def restore_kernels(saved: dict) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    for name, function in saved.items():
        setattr(modeling, name, function)


def flat_gradient(model, input_ids: torch.Tensor, labels: torch.Tensor, loss_fn) -> torch.Tensor:
    """Gradient of the example's loss as one flat float32 vector (detached)."""
    params = [p for _, p in trainable(model)]
    loss = loss_fn(model(input_ids=input_ids).logits, labels)
    return torch.cat([g.reshape(-1).float() for g in torch.autograd.grad(loss, params)]).detach()


def soft_loss(logits: torch.Tensor, probs: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Mean cross-entropy of the next-token distribution against the soft labels, on the completion positions."""
    predicted = torch.log_softmax(logits[0, prompt_length - 1:-1].float(), dim=-1)      # predicts completion tokens
    return -(probs[prompt_length:] * predicted).sum(dim=-1).mean()


def invert(model, embedding: torch.Tensor, target: torch.Tensor, prompt_ids: list[int], completion_length: int, *,
           steps: int, lr: float, temperature: float = 1.0, seed: int = 0, log_every: int = 0) -> dict:
    """Recover `completion_length` tokens. `embedding` is the (V, d) input embedding matrix of the model.
    Returns the recovered ids, the final cosine distance and the distance trace. The model needs double
    backward, so it must be loaded without gradient checkpointing; its train/eval mode is restored."""
    if steps < 1:
        raise ValueError("steps must be at least one")
    device = target.device
    params = [p for _, p in trainable(model)]
    vocab = embedding.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = (torch.randn(completion_length, vocab, generator=generator) * 0.01).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    prompt = torch.tensor(prompt_ids, device=device)
    embedding32 = embedding.float()                  # one float32 copy (V x d), not one per step
    prompt_embeddings = embedding32[prompt]
    trace = []
    was_training = model.training
    model.train()
    try:
        for step in range(steps):
            soft = torch.softmax(z / temperature, dim=-1)
            embeddings = torch.cat([prompt_embeddings, soft @ embedding32], dim=0).unsqueeze(0)
            probs = torch.cat([F.one_hot(prompt, vocab).float(), soft], dim=0)
            loss = soft_loss(model(inputs_embeds=embeddings.to(embedding.dtype)).logits, probs, len(prompt_ids))
            grads = torch.autograd.grad(loss, params, create_graph=True)
            flat = torch.cat([g.reshape(-1).float() for g in grads])
            distance = 1 - F.cosine_similarity(flat, target, dim=0)
            optimizer.zero_grad()
            distance.backward()
            optimizer.step()
            trace.append(float(distance.detach()))
            if log_every and step % log_every == 0:
                print(f"step {step} cosine distance {trace[-1]:.4f}", flush=True)
    finally:
        model.train(was_training)
    return dict(ids=z.detach().argmax(dim=-1).tolist(), distance=trace[-1], trace=trace)


@torch.no_grad()
def _greedy_continuation(model, prompt_ids: list[int], length: int) -> list[int]:
    ids = torch.tensor([prompt_ids], device=next(model.parameters()).device)
    for _ in range(length):
        ids = torch.cat([ids, model(input_ids=ids).logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    return ids[0, len(prompt_ids):].tolist()


@torch.no_grad()
def _candidates(model, prefix: list[int], top_k: int) -> list[int]:
    logits = model(input_ids=torch.tensor([prefix], device=next(model.parameters()).device)).logits[0, -1]
    return logits.float().topk(top_k).indices.tolist()


def coordinate_search(model, target: torch.Tensor, prompt_ids: list[int], completion_length: int, loss_fn, *,
                      top_k: int = 64, passes: int = 3) -> dict:
    """Gradient inversion by coordinate ascent on hard tokens (no double backward, so the fused kernels work).

    Start from the model's own greedy continuation of the prompt (a no-gradient baseline). For each completion
    position, the candidates are the model's top_k next tokens given the current guess of the earlier positions;
    the guess keeps the candidate whose example gradient has the highest cosine with the observed one.
    Returns the ids, the cosine of the starting guess and of the result, per-pass trace and the number of
    gradient evaluations. The attacker knows the model, the adapter, the prompt and the completion length."""
    if top_k < 1 or passes < 1:
        raise ValueError("top_k and passes must be positive")
    device = target.device
    params = [p for _, p in trainable(model)]
    prompt_length = len(prompt_ids)

    def cosine(tokens: list[int]) -> float:
        ids = torch.tensor([prompt_ids + tokens], device=device)
        labels = ids.clone()
        labels[:, :prompt_length] = -100
        loss = loss_fn(model(input_ids=ids).logits, labels)
        flat = torch.cat([g.reshape(-1).float() for g in torch.autograd.grad(loss, params)])
        return float(F.cosine_similarity(flat, target, dim=0))

    model_was_training = model.training
    model.train()
    try:
        guess = _greedy_continuation(model, prompt_ids, completion_length)
        initial, best, evaluations, trace = list(guess), cosine(guess), 1, []
        for _ in range(passes):
            changed = False
            for position in range(completion_length):
                for candidate in _candidates(model, prompt_ids + guess[:position], top_k):
                    if candidate == guess[position]:
                        continue
                    trial = list(guess)
                    trial[position] = candidate
                    score = cosine(trial)
                    evaluations += 1
                    if score > best:
                        best, guess, changed = score, trial, True
            trace.append(best)
            if not changed:
                break
    finally:
        model.train(model_was_training)
    return dict(ids=guess, initial_ids=initial, cosine=best, trace=trace, evaluations=evaluations)
