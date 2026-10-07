# Synthetic LLM smoke data

Small English fixtures for offline pipeline verification only, without a domain-quality claim.
Base splits are grouped by `document_id`; instruction splits by `conversation_id`.
The datasets include train/validation/test and a multi-turn instruction example.

`ppo.jsonl` and `grpo.jsonl` contain 4 train, 2 validation and 2 test prompt/reference
records each. They are independent tiny synthetic fixtures for reward-based online RL.
References are reward targets and never model inputs. See [the RL guide](../../../docs/rl.md).
