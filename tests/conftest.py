import pytest
import torch

from oglodon import Oglodon, OglodonConfig

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_config(**kw) -> OglodonConfig:
    base = dict(
        vocab_size=37,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=48,
        eos_token_id=1,
        pad_token_id=0,
    )
    base.update(kw)
    return OglodonConfig(**base)


@pytest.fixture
def config() -> OglodonConfig:
    return tiny_config()


@pytest.fixture
def model(config) -> Oglodon:
    torch.manual_seed(0)
    return Oglodon(config).eval()


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


# every architectural switch, used by the parametrised model tests
VARIANTS = [
    dict(),
    dict(num_kv_heads=4),  # MHA
    dict(num_kv_heads=1),  # MQA
    dict(use_rope=False),  # learned positions
    dict(use_rmsnorm=False),  # LayerNorm
    dict(use_swiglu=False),  # GELU MLP
    dict(use_rope=False, use_rmsnorm=False, use_swiglu=False, tie_embeddings=False),
]


@pytest.fixture(params=VARIANTS, ids=lambda v: ",".join(f"{k}={x}" for k, x in v.items()) or "default")
def variant_model(request) -> Oglodon:
    torch.manual_seed(0)
    return Oglodon(tiny_config(**request.param)).eval()
