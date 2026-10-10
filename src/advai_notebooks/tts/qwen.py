"""Project-owned native Qwen3-TTS Base teacher forcing, not an external-script runner.

The public token layout follows Qwen's single-speaker finetuning interface:
https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
The Base format is retained; no speaker ID/config conversion or raw reference upload.
"""

import torch


def pack(model, rows):
    config, talker = model.config, model.talker
    device = next(model.parameters()).device
    sequences, targets, codes, masks = [], [], [], []
    for row in rows:
        ids, audio = row["text_ids"].to(device), row["codes"].to(device)
        # Header, four conditioning positions, TTS BOS, body/EOS, codec BOS, codes/EOS.
        text_ids = torch.cat(
            (
                ids[:3],
                ids.new_full((4,), config.tts_pad_token_id),
                ids.new_tensor([config.tts_bos_token_id]),
                ids[3:],
                ids.new_tensor([config.tts_eos_token_id]),
                ids.new_full((len(audio) + 2,), config.tts_pad_token_id),
            )
        )
        codec = ids.new_full(text_ids.shape, talker.config.codec_pad_id)
        codec[3:6] = ids.new_tensor(
            [
                talker.config.codec_nothink_id,
                talker.config.codec_think_bos_id,
                talker.config.codec_think_eos_id,
            ]
        )
        start = len(ids) + 7
        codec[start - 1] = talker.config.codec_bos_id
        codec[start : start + len(audio)] = audio[:, 0]
        codec[start + len(audio)] = talker.config.codec_eos_token_id
        text_embedding = talker.text_projection(talker.get_text_embeddings()(text_ids))
        codec_embedding = talker.get_input_embeddings()(codec)
        codec_embedding[:3] = 0
        codec_embedding[6] = row["speaker_embedding"].to(device)
        embedding = text_embedding + codec_embedding
        for group, layer in enumerate(talker.code_predictor.get_input_embeddings(), 1):
            embedding[start : start + len(audio)] = embedding[start : start + len(audio)] + layer(
                audio[:, group]
            )
        labels = ids.new_full(text_ids.shape, -100)
        labels[start : start + len(audio)] = audio[:, 0]
        labels[start + len(audio)] = talker.config.codec_eos_token_id
        selected = torch.zeros(len(text_ids), device=device, dtype=torch.bool)
        selected[start - 1 : start + len(audio) - 1] = True
        sequences.append(embedding[:-1])
        targets.append(labels[1:])
        masks.append(selected[:-1])
        codes.append(audio)
    length = max(len(sequence) for sequence in sequences)
    attention = torch.stack(
        [torch.arange(length, device=device) < len(sequence) for sequence in sequences]
    )
    return (
        torch.stack(
            [
                torch.nn.functional.pad(sequence, (0, 0, 0, length - len(sequence)))
                for sequence in sequences
            ]
        ),
        torch.stack(
            [
                torch.nn.functional.pad(label, (0, length - len(label)), value=-100)
                for label in targets
            ]
        ),
        attention.long(),
        torch.stack(
            [torch.nn.functional.pad(mask, (0, length - len(mask)), value=False) for mask in masks]
        ),
        torch.cat(codes),
    )


def native_loss(model, rows):
    embedding, labels, attention, mask, codes = pack(model, rows)
    outputs = model.talker(
        inputs_embeds=embedding,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[0][-1][mask]
    residual_logits, _ = model.talker.forward_sub_talker_finetune(codes, hidden)
    # Labels are already next-token aligned. Explicit CE avoids HF's implicit
    # causal shift, including across residual codebooks (which are already aligned).
    first = torch.nn.functional.cross_entropy(outputs.logits.flatten(0, 1), labels.flatten())
    residual = torch.nn.functional.cross_entropy(
        residual_logits.flatten(0, 1), codes[:, 1:].flatten()
    )
    return first + 0.3 * residual
