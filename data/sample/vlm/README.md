# VLM synthetic fixtures

`samples.jsonl`: 8 single-image color QA examples, split train 4 / validation 2 / test 2.
`images/*.ppm`: small RGB fixtures. Paths are relative to the JSONL.
Samples have unique scene groups; decoded images never cross splits.

These deliberately tiny fixtures are versioned via explicit Git ignore exceptions.
They test data contracts and lifecycle mechanics, not model capability.
[Schema and validation](../../../docs/vlm.md).
