import pytest
import yaml

from pqfl.config import load_run_config

GOOD = dict(run_id="t", sites=["A", "B"], clip_bound=1.0, scale_bits=24, modulus_bits=40,
            ring_dim=16384, scaling_mod_size=50, dp_level="none", run_seed=1,
            rekey_every_rounds=None, auth={"method": "none_insecure"})


def write(tmp_path, **changes):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**GOOD, **changes}))
    return path


def test_valid_config_loads_and_hash_depends_on_scheme(tmp_path):
    cfg = load_run_config(write(tmp_path))
    assert cfg.sites == ("A", "B") and cfg.config_hash("A") != cfg.config_hash("B")


@pytest.mark.parametrize("changes", [
    dict(run_id="TODO"), dict(sites=["A"]), dict(sites=["A", "A"]), dict(clip_bound=0),
    dict(clip_bound=float("inf")), dict(run_seed=-1), dict(run_seed=True),
    dict(rekey_every_rounds=0), dict(auth={"method": "unknown"}), dict(scale_bits=None),
])
def test_invalid_config_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_run_config(write(tmp_path, **changes))


def test_shipped_templates():
    assert load_run_config("configs/sim_costs.yaml").auth_method == "mldsa_ephemeral_simulation"
    with pytest.raises(ValueError, match="TODO|missing"):
        load_run_config("configs/experiment.yaml")
