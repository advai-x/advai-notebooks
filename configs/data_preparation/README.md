# Offline preparation

`smoke.yaml` selects judged training negatives and all-corpus evaluation
candidates without loading a model. `bge.yaml` optionally uses an encoder for
hard negatives and top-k candidates. The encoder is owned by this stage only.

`mining.negatives_per_query` controls the prepared row count, not either loss.
`training.batch_size` and `seed` configure preparation execution. The optional
encoder's model/prefix/revision settings belong here. Set returned sample and
candidate paths in the independent task configs. Existing output directories
are never overwritten.
