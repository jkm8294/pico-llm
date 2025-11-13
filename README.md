# pico-llm quick reference

## How to run
```bash
python3 pico-llm.py --block_size 32 --tinystories_weight 0.0 --input_files 3seqs.txt --prompt "0 1 2 3 4" --max_steps_per_epoch 50
```

## Models trained
- `kgram_mlp_seq`: embedding-based k-gram MLP
- `lstm_seq`: single-layer LSTM language model
- `transformer`: RMSNorm Transformer with causal self-attention

## Figures
- Training losses per model: `kgram_mlp_seq_train_loss.png`, `lstm_seq_train_loss.png`, `transformer_train_loss.png`
- Top-p comparison: `top_p_effect.png` (and console samples for greedy / p=0.95 / p=1.0)
- Any optional figures you generate (e.g., `causal_mask.png`) are saved in the repo root alongside the above files

## Notes
- All token histories pass through learned embeddings (even the k-gram model), which keeps the model sizes manageable compared to one-hot encodings.
- Top-p (nucleus) sampling trims the sorted probability tail once the cumulative mass exceeds the target `p`, giving you controllable randomness: e.g., greedy (p=0) is deterministic, 0.95 balances diversity, and 1.0 samples from the full distribution.
