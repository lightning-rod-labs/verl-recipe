"""Wire-to-loss alignment for top-K cross entropy, including the final target."""

import pytest
import torch
from omegaconf import OmegaConf
from tinker.types import Datum, ModelInput, TensorData
from verl_tinker.backends._loss import make_branching_loss
from verl_tinker.data.datum_processing import _datums_to_sft_td

from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize("active_rows", [([False, True, True], [False, True]), ([True, False, True], [True, True])])
@pytest.mark.parametrize("pad_to_multiple", [1, 4])
def test_topk_translation_preserves_weighted_loss_and_final_token(active_rows, pad_to_multiple):
    datums = []
    for active in active_rows:
        rows = len(active)
        targets = torch.arange(rows * 2).reshape(rows, 2) % 7
        weights = torch.tensor([[0.25, 0.75]] * rows) * torch.tensor(active).unsqueeze(-1)
        datums.append(
            Datum(
                model_input=ModelInput.from_ints(list(range(1, rows + 1))),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(targets),
                    "weights": TensorData.from_torch(weights),
                },
            )
        )
    total_weight = sum(d.loss_fn_inputs["weights"].to_torch().sum() for d in datums)
    for datum in datums:
        datum.loss_fn_inputs["weights"] = TensorData.from_torch(
            datum.loss_fn_inputs["weights"].to_torch() / total_weight
        )

    data = _datums_to_sft_td(datums, mini_batch_size=len(datums), pad_to_multiple=pad_to_multiple)
    tu.assign_non_tensor_data(data, "dp_size", 1)
    offsets = data["input_ids"].offsets()
    logits = torch.randn(1, int(offsets[-1]), 7, generator=torch.Generator().manual_seed(42), requires_grad=True)
    loss_fn = make_branching_loss(
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
    processed = loss_fn(student_logits=logits, data=data)
    model_output = {
        key: torch.nested.nested_tensor_from_jagged(value.squeeze(0), offsets) for key, value in processed.items()
    }
    loss, _ = loss_fn(model_output=model_output, data=data)

    expected = logits.sum() * 0
    for i, datum in enumerate(datums):
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights = datum.loss_fn_inputs["weights"].to_torch()
        start = int(offsets[i])
        log_probs = logits[0, start : start + len(targets)].log_softmax(-1)
        expected = expected - (weights * log_probs.gather(-1, targets)).sum()
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    mask = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
    assert (processed["teacher_mass"].squeeze(0) * mask).sum().item() == pytest.approx(1)
    for i, datum in enumerate(datums):
        final_prediction = int(offsets[i]) + datum.model_input.length - 1
        assert actual_grad[0, final_prediction].abs().sum().item() > 0
        assert actual_grad[0, final_prediction + 1].count_nonzero().item() == 0


def test_one_dimensional_sft_keeps_its_causal_mask_alignment():
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "weights": TensorData.from_torch(torch.tensor([0.0, 0.25, 0.75])),
        },
    )
    data = _datums_to_sft_td([datum], mini_batch_size=1)
    tu.assign_non_tensor_data(data, "dp_size", 1)
    loss_fn = make_branching_loss(
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
    log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(log_probs, data["input_ids"].offsets())}
    loss, _ = loss_fn(model_output=model_output, data=data)
    loss.backward()
    assert loss.item() == pytest.approx(2.75)
    torch.testing.assert_close(log_probs.grad, torch.tensor([0.0, -0.25, -0.75, 0.0]))


@pytest.mark.parametrize("world_size,real_count", [(2, 1), (4, 1), (4, 2), (4, 3)])
def test_topk_padding_only_ranks_have_zero_gradient_and_preserve_global_loss(world_size, real_count):
    datums = []
    for index in range(real_count):
        weights = torch.tensor([[0.0, 0.0], [0.25, 0.75], [0.75, 0.25]]) / (2 * real_count)
        datums.append(
            Datum(
                model_input=ModelInput.from_ints([1, 2 + index, 3]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor([[0, 0], [1, 2], [3, 4]])),
                    "weights": TensorData.from_torch(weights),
                },
            )
        )
    data = _datums_to_sft_td(datums, mini_batch_size=real_count, pad_to_multiple=world_size)
    config = OmegaConf.create(
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
    loss_fn = make_branching_loss(config)
    initial_weights = torch.randn(8, 7, generator=torch.Generator().manual_seed(42))
    reference_weights = initial_weights.clone().requires_grad_()
    expected = reference_weights.sum() * 0
    for datum in datums:
        logits = reference_weights[datum.model_input.to_ints()]
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights = datum.loss_fn_inputs["weights"].to_torch()
        expected = expected - (weights * logits.log_softmax(-1).gather(-1, targets)).sum()
    expected.backward()

    rank_losses = []
    rank_gradients = []
    for rank in range(world_size):
        rank_weights = initial_weights.clone().requires_grad_()
        fields = {
            key: torch.nested.as_nested_tensor([data[key].unbind()[rank]], layout=torch.jagged)
            for key in ("input_ids", "position_ids", "loss_mask", "teacher_ids", "teacher_logprobs")
        }
        from tensordict import TensorDict

        local_data = TensorDict(fields, batch_size=[1])
        tu.assign_non_tensor_data(local_data, "dp_size", world_size)
        tu.assign_non_tensor_data(local_data, "__loss_mode__", "topk_distill")
        logits = rank_weights[local_data["input_ids"].values()].unsqueeze(0)
        processed = loss_fn(student_logits=logits, data=local_data)
        outputs = {
            key: torch.nested.nested_tensor_from_jagged(value.squeeze(0), local_data["input_ids"].offsets())
            for key, value in processed.items()
        }
        loss, _ = loss_fn(model_output=outputs, data=local_data)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(rank_weights.grad).all()
        if rank >= real_count:
            assert loss.item() == 0
            assert rank_weights.grad.count_nonzero().item() == 0
        rank_losses.append(loss.detach())
        rank_gradients.append(rank_weights.grad)
    # FSDP averages rank gradients; the token-sum loss compensates with dp_size.
    torch.testing.assert_close(torch.stack(rank_losses).mean(), expected.detach())
    torch.testing.assert_close(torch.stack(rank_gradients).mean(0), reference_weights.grad)
