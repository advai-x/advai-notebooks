# VLA synthetic fixtures

`episodes.jsonl`: 4 instruction-conditioned episodes, each 3 steps, split train 2 / validation 1 / test 1.
Each step has a timestamp, image, 2D state and 2D displacement action.
Control frequency is 10 Hz, units meters, coordinate frame world; horizon defaults to 2.
Colors can repeat within a split, but decoded images never cross splits.

These deliberately tiny fixtures are versioned via explicit Git ignore exceptions.
They test data contracts and lifecycle mechanics, not model capability.
[Schema and validation](../../../docs/vla.md).
