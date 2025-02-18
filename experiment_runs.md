# Experiment Runs
(for deprecated runs which might have appeared in previous versions of the repo, see [old_experiment_runs.md](./old_experiment_runs.md))

## Timestamp mapping
Maps the timestamp in the artifacts folder to the experiment type

### Eureka Experiments
When `max_iterations` is not specified, it is set to the environment's default.

This is a training-only hyperparameter;

NOTE: 2025-02-13 redo franka_cabinet


| timestamp | `env_name` | `max_iterations` |
|-----------|-----------|----------------|
| 2025-02-13_09-26-08 | franka_cabinet | default |
| 2025-02-02_14-49-34 | franka_cabinet | default |
| 2025-02-01_18-54-34 | humanoid | default |
| 2025-01-31_13-55-23 | ant | default |
| 2025-01-30_14-32-06 | ant | 3000 |
| 2025-01-27_20-24-22 | franka_cabinet | 3000 |
| 2025-01-25_04-08-48 | humanoid | 3000 |

### Human Baseline Experiments
Human baselines use the environment's default `max_iterations`.

| timestamp | `env_name` |
|-----------|-----------|
| 2025-01-31_23-25-03 | allegro_hand |
| 2025-01-30_10-48-46 | ant |
| 2025-01-27_01-22-01 | franka_cabinet |
| 2025-01-27_00-07-02 | humanoid |

## Results Comparison
The number after the $\pm$ is the standard deviation. The brackets show the number of epochs used during training.
During evaluation, the environment's default `max_iterations` is used.
As such, only the the max training success for the default epochs can be compared to the final (eval) success.

| `env_name` | Max Training Success (default) | Max Training Success (3000) | Eval Success (default) | Eval Success (3000) | Human Baseline Success | Correlation (default) | Correlation (3000) |
|-----------|--------------------------|-------------------------|-------------------|------------------|-------------------|-------------------|-----------------|
| ant | 8.61 | 11.08 | **8.21** ± 0.47 | 6.68 ± 0.26 | 6.93 ± 0.50 | 0.92 ± 0.01 | 0.97 ± 0.00 |
| franka_cabinet | 0.37 | 0.99 | 0.05 ± 0.06 | **0.09** ± 0.14 | 0.05 ± 0.07 | 0.60 ± 0.19 | 0.77 ± 0.28 |
| humanoid | 6.45 | 8.52 | 5.30 ± 0.65 | 5.60 ± 0.71 | **6.31** ± 0.55 | 1.00 ± 0.00 | 0.98 ± 0.01 |
| allegro_hand | - | - | - | - | 13.11 ± 1.39 | - | - |

