import os
from pathlib import Path
from typing import Optional
import jax
import numpy as np
import polars as pl
import json
import wandb


def save_wandb(
    project: str,
    algo_name: str,
    env_name: str,
    metrics: dict,
    metrics_to_log: Optional[list] = None,
    config: Optional[dict] = None,
):
    """Log metrics to Weights and Biases."""
    if metrics_to_log is None:
        metrics_to_log = list(metrics.keys())

    wandb.login(os.environ.get("WANDB_KEY"))
    wandb.init(
        project=project,
        config=config,
        tags=[algo_name.upper(), f"{env_name.upper()}", f"jax_{jax.__version__}"],
        name=f"{algo_name}_{env_name}",
        mode="online",
    )

    first_metric_key = next(iter(metrics.keys()))
    first_metric = metrics[first_metric_key]
    num_seeds, num_updates = first_metric.shape[0], first_metric.shape[1]

    for update_idx in range(num_updates):
        log_dict = {}

        for run_idx in range(num_seeds):
            run_prefix = f"run_{run_idx}"
            for metric_name in metrics_to_log:
                log_dict[f"{run_prefix}/{metric_name}"] = metrics[metric_name][run_idx][
                    update_idx
                ]

        wandb.log(log_dict)


def save_local(
    algo_name: str,
    env_name: str,
    metrics: dict,
    root_dir: str = "results",
    metrics_to_log: Optional[list] = None,
    config: Optional[dict] = None,
):
    """Save metrics locally in NPZ and Parquet formats."""
    if metrics_to_log is None:
        metrics_to_log = list(metrics.keys())

    output_dir = Path(root_dir) / f"{algo_name}_{env_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_np = jax.tree.map(np.array, metrics)

    np.savez_compressed(output_dir / "metrics.npz", **metrics_np)

    first_metric_key = next(iter(metrics.keys()))
    first_metric = metrics[first_metric_key]
    num_seeds, num_updates = first_metric.shape[0], first_metric.shape[1]

    data = {
        "seed": np.repeat(np.arange(num_seeds), num_updates),
        "update": np.tile(np.arange(num_updates), num_seeds),
    }

    # Add all scalar metrics to the data dictionary
    skipped_metrics = []
    for metric_name in metrics_to_log:
        metric_array = np.array(
            [metrics[metric_name][run_idx] for run_idx in range(num_seeds)]
        )
        if metric_array.ndim == 2 and metric_array.shape == (num_seeds, num_updates):
            data[metric_name] = metric_array.flatten()
        else:
            skipped_metrics.append((metric_name, metric_array.shape))

    # Create and save Polars DataFrame
    df = pl.DataFrame(data)
    df.write_parquet(output_dir / "metrics.parquet")

    if config is not None:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Print summary
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - metrics.npz (full metrics)")
    print(f"  - metrics.parquet ({len(df):,} rows, {len(data) - 2} metrics)")
    if skipped_metrics:
        print(f"\n  Skipped {len(skipped_metrics)} non-scalar metrics:")
        for name, shape in skipped_metrics:
            print(f"    - {name}: shape {shape}")
