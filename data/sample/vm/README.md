# VM synthetic fixtures

`episodes.jsonl`: 6 numerical episodes, split train 3 / validation 2 / test 1.
Each has 4 observations, 3 aligned actions and 4 timestamps at 10 Hz.
Dynamics are `next_state = state + action`; states/actions are 2D and use meters.

These deliberately tiny fixtures are versioned via explicit Git ignore exceptions.
They test data contracts and lifecycle mechanics, not model capability.
[Schema and validation](../../../docs/vm.md).
