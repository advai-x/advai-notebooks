# Synthetic diffusion fixtures

Eight original 16x16 RGB color gradients, stored as small text PPM files. No real user images.
`samples.jsonl` uses image paths relative to itself, captions, IDs, group IDs and explicit
train 4 / validation 2 / test 2 splits. All groups and decoded images are split-disjoint.
They verify data integrity and the training/export/inference lifecycle, not image quality.
Keep real images outside this versioned fixture directory.
