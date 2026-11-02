"""Final-campaign job construction: the DP randomness choice must reach every DP job."""
import importlib.util
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location("submit_final", Path("scripts/submit_final.py"))
submit_final = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(submit_final)

SPEC = dict(rounds=4, lr=2e-4, micro_batch=4, logical_batch=32, clamp_bound=0.02, clip=4.0, delta=1e-5,
            seeds=[1, 2], dp_levels={"none": None, "eps8": 8.0},
            configs=[{"scheme": "A", "levels": ["none", "eps8"]}, {"scheme": "Z", "levels": ["none"]}])
COMMON = {k: k for k in ("model_path", "rows_path", "validation_path", "holdout_path", "manifest_path",
                         "canaries_path", "base_nll_path")}


def jobs(**spec):
    return dict(submit_final.build_jobs({**SPEC, **spec}, COMMON, [1], out_root="runs/final", oqs_path="/oqs"))


@pytest.mark.parametrize("value", [True, False])
def test_dp_jobs_carry_the_declared_randomness(value):
    built = jobs(secret_randomness=value)
    assert built["final-A-eps8-s1"]["secret_randomness"] is value
    assert "secret_randomness" not in built["final-A-none-s1"]          # pilot_run refuses it without DP
    assert "secret_randomness" not in built["final-Z-none-s1"]


@pytest.mark.parametrize("value", [None, "TODO", 1])
def test_an_undeclared_choice_is_refused(value):
    spec = {**SPEC, "secret_randomness": value} if value is not None else dict(SPEC)
    with pytest.raises(ValueError, match="secret_randomness"):
        dict(submit_final.build_jobs(spec, COMMON, [1], out_root="runs/final", oqs_path="/oqs"))


def test_the_shipped_specs_declare_it_or_stay_blocked():
    for path in ("configs/final_campaign.yaml", "configs/second_model_campaign.yaml"):
        spec = yaml.safe_load(open(path, encoding="utf-8"))
        if "secret_randomness" in spec and isinstance(spec["secret_randomness"], bool):
            continue
        with pytest.raises(ValueError, match="secret_randomness"):
            dict(submit_final.build_jobs(spec, COMMON, spec["seeds"][:1], out_root="r", oqs_path="/oqs"))
