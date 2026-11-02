"""Needs torch: run in the training environment with `--noconftest`."""
import pytest

torch = pytest.importorskip("torch")

from pqfl.attacks.gradient_inversion import flat_gradient, invert, soft_loss  # noqa: E402
from pqfl.attacks.inversion import bag_recall, exceeds_null, null_scores  # noqa: E402
from pqfl.train.dpsgd import example_loss_sum  # noqa: E402

VOCAB, DIM = 12, 16


class ToyLora(torch.nn.Module):
    """Frozen embedding and head; trainable A and B on top, accepting ids or embeddings."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, DIM)
        self.a = torch.nn.Linear(DIM, 6, bias=False)
        self.b = torch.nn.Linear(6, DIM, bias=False)
        self.head = torch.nn.Linear(DIM, VOCAB, bias=False)
        torch.nn.init.normal_(self.b.weight, std=0.5)
        for module in (self.emb, self.head):
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids=None, inputs_embeds=None):
        h = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        mixed = torch.cumsum(h, dim=1)                     # tokens interact (a causal mix), so order matters
        out = self.head(h + self.b(torch.tanh(self.a(mixed))))
        return type("Out", (), {"logits": out})()


def target_gradient(model, ids, prompt_length):
    labels = ids.clone()
    labels[:, :prompt_length] = -100
    return flat_gradient(model, ids, labels, example_loss_sum), labels


def test_soft_loss_with_one_hot_labels_equals_the_hard_loss():
    model = ToyLora()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = ids.clone(); labels[:, :3] = -100
    logits = model(input_ids=ids).logits
    probs = torch.nn.functional.one_hot(ids[0], VOCAB).float()
    assert float(soft_loss(logits, probs, 3).detach()) == pytest.approx(float(example_loss_sum(logits, labels).detach()), rel=1e-5)


def test_the_attack_recovers_a_short_completion_on_a_tiny_model():
    model = ToyLora()
    ids = torch.tensor([[1, 2, 3, 7, 5, 9, 4, 8]])
    target, _ = target_gradient(model, ids, prompt_length=3)
    out = invert(model, model.emb.weight.detach(), target, ids[0, :3].tolist(), completion_length=5, steps=400,
                 lr=0.3, temperature=0.5, seed=1)
    truth = ids[0, 3:].tolist()
    assert out["distance"] < 0.2 and out["trace"][-1] < out["trace"][0]
    null = null_scores(truth, bag_recall, list(range(VOCAB)), length=5, draws=300, seed=2)
    assert bag_recall(truth, out["ids"]) >= 0.8 and exceeds_null(bag_recall(truth, out["ids"]), null, 0.95)


def test_the_recovered_tokens_follow_the_observed_gradient_not_a_fixed_guess():
    # A low cosine distance alone proves little (soft logits can match many directions), so what counts is
    # that the tokens track the sequence behind the observed gradient: invert two gradients, get two answers.
    model = ToyLora()
    first = torch.tensor([[1, 2, 3, 7, 5, 9, 4, 8]])
    second = torch.tensor([[1, 2, 3, 10, 11, 0, 6, 2]])
    results = {}
    for name, ids in (("first", first), ("second", second)):
        target, _ = target_gradient(model, ids, 3)
        results[name] = invert(model, model.emb.weight.detach(), target, ids[0, :3].tolist(), completion_length=5,
                               steps=400, lr=0.3, temperature=0.5, seed=1)["ids"]
    truth = {"first": first[0, 3:].tolist(), "second": second[0, 3:].tolist()}
    for name, other in (("first", "second"), ("second", "first")):
        assert bag_recall(truth[name], results[name]) > bag_recall(truth[name], results[other])



def test_invert_restores_the_mode_and_rejects_zero_steps():
    model = ToyLora().eval()
    ids = torch.tensor([[1, 2, 3, 7, 5]])
    target, _ = target_gradient(model, ids, prompt_length=3)
    invert(model, model.emb.weight.detach(), target, [1, 2, 3], completion_length=2, steps=2, lr=0.1)
    assert model.training is False
    with pytest.raises(ValueError):
        invert(model, model.emb.weight.detach(), target, [1, 2, 3], completion_length=2, steps=0, lr=0.1)


def test_coordinate_search_recovers_tokens_with_a_selective_objective():
    from pqfl.attacks.gradient_inversion import coordinate_search
    model = ToyLora()
    ids = torch.tensor([[1, 2, 3, 7, 5, 9, 4, 8]])
    target, _ = target_gradient(model, ids, prompt_length=3)
    out = coordinate_search(model, target, ids[0, :3].tolist(), 5, example_loss_sum, top_k=VOCAB, passes=4)
    assert out["cosine"] > 0.999 and out["ids"] == ids[0, 3:].tolist()
    assert out["cosine"] >= out["trace"][0] - 1e-9 and out["evaluations"] > 1
    with pytest.raises(ValueError):
        coordinate_search(model, target, [1, 2, 3], 5, example_loss_sum, top_k=0)
