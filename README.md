# Eureka artifacts
Outputs of attempts to reproduce Eureka (Ma et. al. 2023).

Currently done for the Ant, Humanoid, and FrankaCabinet tasks.
(rest are WIP)

## Other links
- [Experiment Runs](./experiment_runs.md) to map the folders in this repo to the experiment names

## Navigation
- [Ant Task](#ant-task)
- [Humanoid Task](#humanoid-task)
- [FrankaCabinet Task](#frankacabinet-task)
- [AllegroHand Task](#allegrohand-task) (WIP)
- [ShadowHand Pen Spinning Demo](#shadowhand-pen-spinning-demo)

## Quick summary
- Ant: Training with default epochs beat human baseline
- Humanoid: Does not beat human baseline, unlike in the paper
- FrankaCabinet: Training with epochs=3000 beats human baseline, but default does not

## Ant Task

### Performance Plot
![Ant Performance](plots/Ant_comparison.png)

### Evaluation Results
| Default Epochs | 3000 Epochs | Human Baseline |
|:-------------:|:-----------:|:--------------:|
| ![Default Epochs](gifs/AntGPT_epoch__eval.gif) | ![3000 Epochs](gifs/AntGPT_epoch_3000_eval.gif) | ![Human Baseline](gifs/Ant_eval.gif) |

### Training Progress

#### Default Epochs Training Progress (Iterations 0-4)
| Iter0 | Iter1 | Iter2 | Iter3 | Iter4 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![Iter0](gifs/AntGPT_epoch__iter0.gif) | ![Iter1](gifs/AntGPT_epoch__iter1.gif) | ![Iter2](gifs/AntGPT_epoch__iter2.gif) | ![Iter3](gifs/AntGPT_epoch__iter3.gif) | ![Iter4](gifs/AntGPT_epoch__iter4.gif) |

#### 3000 Epochs Training Progress (Iterations 0-4)
| Iter0 | Iter1 | Iter2 | Iter3 | Iter4 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![Iter0](gifs/AntGPT_epoch_3000_iter0.gif) | ![Iter1](gifs/AntGPT_epoch_3000_iter1.gif) | ![Iter2](gifs/AntGPT_epoch_3000_iter2.gif) | ![Iter3](gifs/AntGPT_epoch_3000_iter3.gif) | ![Iter4](gifs/AntGPT_epoch_3000_iter4.gif) |

## Humanoid Task

### Performance Plot
![Humanoid Performance](plots/Humanoid_comparison.png)

### Evaluation Results
| Default Epochs | 3000 Epochs | Human Baseline |
|:-------------:|:-----------:|:--------------:|
| ![Default Epochs](gifs/HumanoidGPT_epoch__eval.gif) | ![3000 Epochs](gifs/HumanoidGPT_epoch_3000_eval.gif) | ![Human Baseline](gifs/Humanoid_eval.gif) |

### Training Progress

#### Default Epochs Training Progress (Iterations 0-4)
| Iter0 | Iter1 | Iter2 | Iter3 | Iter4 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![Iter0](gifs/HumanoidGPT_epoch__iter0.gif) | ![Iter1](gifs/HumanoidGPT_epoch__iter1.gif) | ![Iter2](gifs/HumanoidGPT_epoch__iter2.gif) | ![Iter3](gifs/HumanoidGPT_epoch__iter3.gif) | ![Iter4](gifs/HumanoidGPT_epoch__iter4.gif) |

#### 3000 Epochs Training Progress (Iterations 0-4)
| Iter0 | Iter1 | Iter2 | Iter3 | Iter4 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![Iter0](gifs/HumanoidGPT_epoch_3000_iter0.gif) | ![Iter1](gifs/HumanoidGPT_epoch_3000_iter1.gif) | ![Iter2](gifs/HumanoidGPT_epoch_3000_iter2.gif) | ![Iter3](gifs/HumanoidGPT_epoch_3000_iter3.gif) | ![Iter4](gifs/HumanoidGPT_epoch_3000_iter4.gif) |

## FrankaCabinet Task

### Performance Plot
![FrankaCabinet Performance](plots/FrankaCabinet_comparison.png)

### Evaluation Results
| Default Epochs | 3000 Epochs | Human Baseline |
|:-------------:|:-----------:|:--------------:|
| ![Default Epochs](gifs/FrankaCabinetGPT_epoch__eval.gif) | ![3000 Epochs](gifs/FrankaCabinetGPT_epoch_3000_eval.gif) | ![Human Baseline](gifs/FrankaCabinet_eval.gif) |

### Training Progress

#### Default Epochs Training Progress (Iterations 0-4)
| Iter0 | Iter1 | Iter2 | Iter3 | Iter4 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![Iter0](gifs/FrankaCabinetGPT_epoch__iter0.gif) | ![Iter1](gifs/FrankaCabinetGPT_epoch__iter1.gif) | ![Iter2](gifs/FrankaCabinetGPT_epoch__iter2.gif) | ![Iter3](gifs/FrankaCabinetGPT_epoch__iter3.gif) | ![Iter4](gifs/FrankaCabinetGPT_epoch__iter4.gif) |

#### 3000 Epochs Training Progress (Iterations 0-4)
| Iter0 | Iter1 | Iter2 | Iter3 | Iter4 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![Iter0](gifs/FrankaCabinetGPT_epoch_3000_iter0.gif) | ![Iter1](gifs/FrankaCabinetGPT_epoch_3000_iter1.gif) | ![Iter2](gifs/FrankaCabinetGPT_epoch_3000_iter2.gif) | ![Iter3](gifs/FrankaCabinetGPT_epoch_3000_iter3.gif) | ![Iter4](gifs/FrankaCabinetGPT_epoch_3000_iter4.gif) |

## AllegroHand Task

### Performance Plot
![AllegroHand Performance](plots/AllegroHand_comparison.png)

### Evaluation Results
![Human Baseline](gifs/AllegroHand_eval.gif)

## ShadowHand Pen Spinning Demo
This is a visualization of the pre-trained pen spinning demo, rather than a result from training.

![Original Demo](gifs/original_demo_ShadowHandSpin.gif)

