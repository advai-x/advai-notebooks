# LLM PPO and GRPO

English | [简体中文](rl.zh-CN.md)

PPO and GRPO are independent online reinforcement-learning tasks in
`src/advai_notebooks/llm/ppo/` and `src/advai_notebooks/llm/grpo/`.
They do not import Base, Instruct or other tasks. Project-owned PyTorch loops handle
training; native HF causal LMs and PEFT handle networks and serialization.
Instruct remains SFT; these tracks do not change its training method.

## Run

```bash
python -m pip install -e '.[llm-ppo,llm-grpo,dev]'
python -m advai_notebooks.llm.ppo validate --config configs/llm/ppo/smoke.yaml
python -m advai_notebooks.llm.ppo run --config configs/llm/ppo/smoke.yaml
python -m advai_notebooks.llm.grpo run --config configs/llm/grpo/smoke.yaml
```

Aliases: `advai-llm-ppo`, `advai-llm-grpo`. Each task provides `smoke.yaml`,
`lora.yaml` and `hf.yaml`; the last requires your data and is not a verified domain
training or GPU-performance result. `model.id` accepts an HF model or local
full/merged export, including Instruct SFT artifacts. Merge adapter-only models
before using them as initialization. This file-level handoff is optional, not a
module dependency.

## Data and rewards

Every JSONL record contains either `prompt` or `messages`, never both. Messages may
start with system, then alternate user/assistant and end with user. `reference`
is used for rewards, not prompt rendering or teacher forcing.

```json
{"id":"math-1","group_id":"problem-1","split":"train","messages":[{"role":"user","content":"compute 1 and 1"}],"reference":"2","metadata":{"domain":"math"}}
```

Train/validation/test are required. Duplicate IDs and identical prompts/groups
across splits are rejected. Assign paraphrases and related tasks the same `group_id`.
Prompts exceeding the context budget fail rather than truncate. The budget is
prompt tokens + max(training rollout budget, evaluation generation budget).

Each task's `reward.py` provides weighted exact match, token-overlap F1, JSON
validity and target-length proximity. Exact match/overlap lowercase and normalize
whitespace, and need nonempty references. Length counts whitespace-separated words
in decoded text; these rules suit smoke tests/simple tasks, not every domain.
There is no implicit reward model. A custom scalar callback may use rules, a judge
or a trained reward model:

```yaml
reward:
  exact_match_weight: 0.0
  custom_function: my_rewards:score
```

```python
# my_rewards.py must be importable. Return a deterministic, finite float.
def score(row: dict, completion: str) -> float:
    return float(completion.strip() == row["reference"].strip())
```

Custom rewards are added to enabled built-in terms and may be negative; they are
not supervised-label cross-entropy. Callbacks should read only the current
row/completion, not held-out data. Pin any callback models, dependencies and services.

## Algorithm differences

| Component | PPO | GRPO |
| --- | --- | --- |
| Policy | Configurable causal LM, full or LoRA | Configurable causal LM, full or LoRA |
| Critic | Linear value head sharing actor decoder features | No value head |
| Samples per prompt | One completion per rollout | `group_size` completions, at least two |
| Advantage | Token-level GAE; optional normalization | Within-prompt reward centering and population-standard-deviation scaling |
| Policy objective | Token-ratio clipped surrogate | Token-ratio clipped surrogate |
| Value objective | Clipped value regression | None |
| Reference KL | Sampled log ratio subtracted from token rewards | `expm1(ref_logp - logp) - (ref_logp - logp)` added to loss |
| Zero reward variance | GAE still depends on critic/reference | Zero group advantage; KL still applies |

PPO shares the actor backbone with its critic, so value regression can update
trainable actor parameters. It does not duplicate a full value model. LoRA updates
only adapters and the value head. Values use pre-action hidden representations.
Terminal task rewards are assigned to the last valid completion token. EOS and
reaching the generation budget both terminate the episode with zero value bootstrap.
These conventions and critic changes belong to PPO alone.

Both tasks freeze the initial reference policy. Each batch generates fresh samples,
caches old-policy log probabilities, performs `update_epochs` optimization passes
and discards the rollout. GRPO computes group statistics before minibatch splitting.
Loss averages valid tokens within each completion, then averages completions,
keeping weights independent of response length. `gradient_accumulation` accumulates
optimization minibatches within that rollout; incomplete windows use actual sample counts.

Training uses temperature-scaled, full-vocabulary categorical sampling with dropout
disabled and no top-k/top-p filtering, matching sampling and update log probabilities.
`algorithm.temperature/max_new_tokens` control training sampling; `generation.*`
controls public inference. Evaluation is greedy with `generation.max_new_tokens`.
Shared `llm/rollout.py` contains only sampling, position IDs, log probabilities and
EOS masks, not rewards, advantages, losses, metrics or selection.

Algorithm references: [PPO](https://arxiv.org/abs/1707.06347),
[GRPO in DeepSeekMath](https://arxiv.org/abs/2402.03300).
This implementation is outcome-supervised GRPO with population standard deviation;
it has no process rewards, iterative reference updates or other GRPO variants.

## Evaluation, persistence and recovery

Lifecycle: data/budget validation → baseline validation/test → online sampling and
updates → per-epoch validation → maximize validation `reward_mean` → held-out test →
export. Test never updates parameters or selects models. Epoch zero can win; ties
select the later epoch. Tiny random models and synthetic rewards check mechanics,
not improved reasoning ability.

```text
outputs/llm/{ppo,grpo}/smoke/
  config.json environment.json report.json
  latest.json
  checkpoints/checkpoint-N/
    policy/                  # Native actor/tokenizer/metadata; PPO also has a value head
    reference/               # Frozen initial reference and loading contract
    state.pt                 # Optimizer and Python/NumPy/Torch/CUDA/MPS RNG
    resume.json              # Epoch/history/selection and provenance signatures
  export/                    # Validation-selected inference model; possibly epoch zero
  latest_export/             # Model after the final training epoch
```

Full exports support native `AutoModelForCausalLM.from_pretrained` actor loading.
PPO separately saves `value_head.safetensors`; generation does not use it, while
the task facade reloads it. LoRA exports record a pinned Hub base or include a
local `base_model/` snapshot. Native PEFT requires loading a bundled base explicitly.
Both tasks support `export --merged`, safely merging a copy into a standalone HF actor.

```bash
python -m advai_notebooks.llm.ppo run --config configs/llm/ppo/smoke.yaml \
  --resume outputs/llm/ppo/smoke/latest
python -m advai_notebooks.llm.grpo evaluate --config configs/llm/grpo/smoke.yaml \
  --checkpoint outputs/llm/grpo/smoke/export --split test \
  --output outputs/llm/grpo/smoke/test.json
```

Resume only from the original run's latest complete epoch checkpoint, not an
inference export. Checks cover configuration (including planned epochs), data,
task source, environment, tokenizer/template, actual initial weights and the custom
reward module's source. Keep the original epoch setting when resuming.
Checkpoints are staged in `.incomplete` directories before atomic commit. Inspect
and remove a leftover incomplete directory before replaying that epoch.
Learning rate is constant, so there is no scheduler state; optimizer and RNG state
are saved. Distributed launches are rejected: do not use torchrun to write multiple
replicas into one output directory.

## Inference and publication

```python
from advai_notebooks.llm.ppo.model import PPOModel
from advai_notebooks.llm.grpo.model import GRPOModel

ppo = PPOModel.load("outputs/llm/ppo/smoke/export", device="cpu")
grpo = GRPOModel.load("outputs/llm/grpo/smoke/export", device="cpu")
print(ppo.chat([{"role": "user", "content": "compute 1 and 1"}]))
print(grpo.generate(["hello world"]))
```

`generate` accepts a text list; `chat` accepts one messages array. CLIs support
generate/chat, Hub revision/cache options and export/publish.
`publish --dry-run` validates the English model card and package offline.
Packages exclude optimizer, reference checkpoints, raw data and logs; PPO includes
its small value head for complete facade loading.

## Implementation limits

Supported: single-device CPU/MPS/CUDA fp32, full/LoRA, accumulation/clipping,
custom terminal rewards, native exports and epoch recovery. CPU smoke behavior
does not establish CUDA/MPS correctness or performance. fp16/bf16, gradient
checkpointing, QLoRA, multi-GPU, asynchronous/vLLM rollouts and long-context
performance optimizations are unsupported and rejected where configurable.
Sampling uses a full-sequence forward per token, prioritizing readability and
algorithm checks; evaluate large-model memory and throughput separately.
Each task retains its own metrics, losses, rewards, sampling organization and trainer.

## Recorded demo releases

The recorded private demos start from existing tiny Instruct SFT weights and run
two full-training epochs: eight PPO optimizer updates and 32 GRPO updates.
Both selected epoch two and used English model cards.

| Task | Pinned model SHA | Pinned Dataset SHA |
| --- | --- | --- |
| [PPO](https://huggingface.co/widebluesky/wbs-llm-ppo-demo) | `25beb2018bbd7a493a36ef5a5793c4ca149b2dec` | `5b621b1b02f8f097a2036c0aa0f915e4df5b8e31` |
| [GRPO](https://huggingface.co/widebluesky/wbs-llm-grpo-demo) | `fdbe74d2c76db0a7c43d432af70a5839e87442e7` | `3246400626f9c419fd1eaa4b36a1594b53089be2` |

The release record documents `demo-v1` model/Dataset tags, verified data associations
and record fingerprints, matching downloaded file hashes, native/facade logits,
CLI/API generate/chat outputs and test reports. Both test exact match and mean reward
are 0.0: these are engineering demos, not effective answer models or evidence of
RL improvements. See the [recorded release](releases/20261008-rl-demo-v1.json)
for the full validation history; these are historical records, not a fresh Hub audit.
