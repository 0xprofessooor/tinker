# Stock Prediction

A quantile-based state prediction model for financial asset price dynamics, trained on simulated GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) environments.

The model learns to predict both the conditional distribution of future returns (via quantile regression) and conditional volatility (via variance prediction) for multi-asset portfolios. It captures non-stationary volatility clustering — a hallmark of real financial time series.

To train the model:

```bash
uv run -m tinker.stock_pred.run
```

To train and run the full evaluation pipeline with visualizations:

```bash
uv run -m tinker.stock_pred.eval
```

## Architecture

```
(obs, obs², action) ──> GRU Cell ──> h_next
                                       │
                        ┌──────────────┼──────────────┐
                        v              │              v
                    Encoder            │        Volatility Head
                 h,a → Linear(256)     │        h → Linear(128)
                    → LayerNorm        │           → SiLU
                    → SiLU             │           → Linear(obs_dim)
                    → Linear           │           → softplus
                        │              │              │
                        v              │              v
                   ψ (embeddings)      │         pred_var
                        │              │
            τ ──> Cosine Basis ──> φ   │
                        │              │
                    ψ · φ (modulate)   │
                        │              │
                    Decoder            │
                 → Linear(256)         │
                    → SiLU             │
                    → Linear(1)        │
                        │              │
                        v              v
                  base_quantiles    std scaling
                        │              │
                        └──────┬───────┘
                               v
                     Quantile predictions
```

## Key Techniques

- **Quantile regression with cosine basis**: A single network predicts the full conditional quantile function by modulating embeddings with τ-dependent cosine features. Trained via pinball loss.
- **Volatility head**: Separate output predicts conditional variance via softplus, trained with Gaussian negative log-likelihood.
- **Squared observation features**: Input includes obs² alongside obs to the GRU, capturing ARCH-type volatility clustering effects.
- **GRU recurrence**: Maintains temporal context across time steps; hidden state resets on episode boundaries.
- **Experience replay**: Uses FlashBax flat buffer for sampling past transitions with stored RNN hidden states.
- **Dual loss**: `total_loss = pinball_loss + vol_loss_coeff * volatility_loss`.

## Evaluation Metrics

The `eval.py` pipeline computes and visualizes:

- **Median-prediction RMSE**: Accuracy of the τ=0.50 quantile prediction.
- **Volatility tracking**: Pearson and Spearman correlation between predicted 90%-interval width and rolling realized volatility.
- **Quantile calibration**: Empirical coverage at each τ level vs ideal (should match τ).
- **Mean calibration error**: MAE between empirical and ideal quantile coverage.

Output saved to `results/thinker_garch_eval.png`.

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `num_steps` | 100K | Total environment steps |
| `train_freq` | 1000 | Steps between updates |
| `buffer_size` | 10K | Replay buffer capacity |
| `batch_size` | 64 | Training batch size |
| `num_epochs` | 10 | Epochs per update |
| `lr` | 3e-4 | Adam with ε=1e-12 |
| `max_grad_norm` | 1.0 | Gradient clip norm |
| `vol_loss_coeff` | 1.0 | Volatility loss weight |
| `embedding_dim` | 64 | Cosine basis embedding size |
| `rnn_hidden_dim` | 128 | GRU hidden state size |

## GARCH Environment

Default assets:

| Asset | μ (drift) | ω (intercept) | α (shock) | β (persistence) |
|---|---|---|---|---|
| AAPL | 5e-4 | 1e-5 | 0.05 | 0.9 |
| BTC | 1.5e-3 | 1e-4 | 0.15 | 0.8 |
