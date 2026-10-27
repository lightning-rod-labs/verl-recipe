"""Data-parallel padding must preserve native Tinker loss and gradients."""

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tinker.types import Datum, ModelInput, TensorData
from verl_tinker.backends._loss import make_branching_loss
from verl_tinker.data.datum_processing import _datums_to_update_actor_td

from verl.utils import tensordict_utils as tu


def _loss_fn():
    return make_branching_loss(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "_target_": "verl.workers.config.actor.FSDPActorConfig",
                        "strategy": "fsdp",
                        "rollout_n": 1,
                        "ppo_micro_batch_size_per_gpu": 1,
                    }
                }
            }
        )
    )


def _rank_batch(data, rank, world_size, loss_spec):
    fields = {}
    for key in (
        "input_ids",
        "position_ids",
        "prompts",
        "responses",
        "attention_mask",
        "response_mask",
        "old_log_probs",
        "advantages",
        "loss_mask",
    ):
        value = data[key]
        fields[key] = (
            torch.nested.as_nested_tensor([value.unbind()[rank]], layout=torch.jagged)
            if value.is_nested
            else value[rank : rank + 1]
        )
    local = TensorDict(fields, batch_size=[1])
    tu.assign_non_tensor(
        local,
        dp_size=world_size,
        batch_num_tokens=float(data["loss_mask"].sum()),
        global_batch_size=len(data),
        __tinker_loss_spec__=loss_spec,
    )
    return local


def _model_output(weights, data):
    tokens = data["input_ids"].values()
    labels = torch.roll(tokens, shifts=-1)
    log_probs = weights[tokens].log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return {"log_probs": torch.nested.nested_tensor_from_jagged(log_probs, data["input_ids"].offsets())}


@pytest.mark.parametrize("world_size,real_count", [(2, 1), (4, 1), (4, 3)])
@pytest.mark.parametrize(
    "loss_spec",
    [{"name": "importance_sampling"}, {"name": "ppo", "clip_ratio_low": 0.2, "clip_ratio_high": 0.2}],
    ids=["importance_sampling", "ppo"],
)
def test_rl_padding_only_ranks_preserve_global_loss_and_gradient(world_size, real_count, loss_spec):
    datums = [
        Datum(
            model_input=ModelInput.from_ints([1, 2 + index, 3]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor([2 + index, 3, 4])),
                "logprobs": TensorData.from_torch(torch.tensor([0.0, -1.0, -1.0])),
                "advantages": TensorData.from_torch(torch.tensor([0.0, 0.5, -0.25])),
            },
        )
        for index in range(real_count)
    ]
    loss_fn = _loss_fn()
    reference_data = _datums_to_update_actor_td(datums, mini_batch_size=real_count)
    tu.assign_non_tensor(
        reference_data,
        dp_size=1,
        batch_num_tokens=float(reference_data["loss_mask"].sum()),
        global_batch_size=real_count,
        __tinker_loss_spec__=loss_spec,
    )
    initial = torch.randn(8, 8, generator=torch.Generator().manual_seed(42))
    reference_weights = initial.clone().requires_grad_()
    expected, _ = loss_fn(model_output=_model_output(reference_weights, reference_data), data=reference_data)
    expected.backward()

    padded = _datums_to_update_actor_td(datums, mini_batch_size=real_count, pad_to_multiple=world_size)
    losses = []
    gradients = []
    for rank in range(world_size):
        local = _rank_batch(padded, rank, world_size, loss_spec)
        weights = initial.clone().requires_grad_()
        loss, _ = loss_fn(model_output=_model_output(weights, local), data=local)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(weights.grad).all()
        if rank >= real_count:
            assert loss.item() == 0
            assert weights.grad.count_nonzero().item() == 0
        losses.append(loss.detach())
        gradients.append(weights.grad)
    torch.testing.assert_close(torch.stack(losses).mean(), expected.detach())
    torch.testing.assert_close(torch.stack(gradients).mean(0), reference_weights.grad)


def test_globally_empty_importance_sampling_mask_still_rejected():
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3])),
            "logprobs": TensorData.from_torch(torch.tensor([0.0, -1.0])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, 1.0])),
        },
    )
    padded = _datums_to_update_actor_td([datum], mini_batch_size=1, pad_to_multiple=2)
    local = _rank_batch(padded, 1, 2, {"name": "importance_sampling"})
    tu.assign_non_tensor_data(local, "batch_num_tokens", 0)
    weights = torch.randn(8, 8, requires_grad=True)
    with pytest.raises(ValueError, match="response_mask must contain at least one valid token"):
        _loss_fn()(model_output=_model_output(weights, local), data=local)
