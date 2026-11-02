"""Needs torch: runs in the training environment (`pytest --noconftest tests/test_train_local.py`)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pqfl.train.local import completion_loss, count_trainable, flatten, load_flat, train_local  # noqa: E402


class CharTokenizer:
    def encode(self, text):
        return [2 + ord(c) % 40 for c in text]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(50, 16)
        self.lin = torch.nn.Linear(16, 50)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.lin(self.emb(input_ids)))


ITEMS = [dict(id=str(i), prompt=f"Q{i}: pick Answer:", completion=" A" if i % 2 else " B") for i in range(8)]
KW = dict(epochs=3, lr=1e-2, max_len=32, eos_id=1, pad_id=0, seed=5)


def test_flatten_and_load_roundtrip_and_length_check():
    model = Tiny()
    vector = flatten(model)
    assert vector.dtype == torch.float32 and vector.numel() == count_trainable(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    load_flat(model, vector)
    assert torch.equal(flatten(model), vector)
    with pytest.raises(ValueError):
        load_flat(model, vector[:-1])


def test_loss_counts_only_completion_tokens():
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[-100, -100, 3, 1]])           # positions 2 and 3 are scored
    total, count = completion_loss(logits, labels)
    assert count == 2 and float(total) == pytest.approx(2 * torch.log(torch.tensor(5.0)).item(), rel=1e-5)
    total0, count0 = completion_loss(logits, torch.full((1, 4), -100))
    assert count0 == 0 and float(total0) == 0.0


def test_training_reduces_the_loss_and_is_deterministic():
    a = train_local(Tiny(), ITEMS, CharTokenizer(), micro_batch=2, accumulation=2, **KW)
    b = train_local(Tiny(), ITEMS, CharTokenizer(), micro_batch=2, accumulation=2, **KW)
    assert a["losses"] == b["losses"] and a["steps"] == 3 * 2
    assert a["losses"][-1] < a["losses"][0] and a["tokens"] > 0


def test_accumulation_equals_one_large_batch():
    small = train_local(Tiny(), ITEMS, CharTokenizer(), micro_batch=1, accumulation=4, **KW)
    large = train_local(Tiny(), ITEMS, CharTokenizer(), micro_batch=4, accumulation=1, **KW)
    assert small["steps"] == large["steps"]
    for x, y in zip(small["losses"], large["losses"]):
        assert x == pytest.approx(y, rel=1e-4)
