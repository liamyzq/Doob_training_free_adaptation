# Training-Free Adaptation of Diffusion Models via Doob’s $h$-Transform

This is the implementation of Training-Free Adaptation of Diffusion Models via Doob’s $h$-Transform for locomotion datasets.

The code base is adapted from:
- [Inference-time Scaling of Diffusion Models through Classical Search](https://github.com/XiangchengZhang/Diffusion-inference-scaling)
- [CEP-energy-guided-diffusion](https://github.com/thu-ml/CEP-energy-guided-diffusion)

## Requirements

Install:
- [PyTorch](https://pytorch.org/)
- [MuJoCo](https://github.com/deepmind/mujoco)
- [D4RL](https://github.com/Farama-Foundation/D4RL)

## Pretrained Models

Download checkpoints from [this link](https://drive.google.com/drive/folders/1snFcmcJaalcCWW9roBjeCjpWjpCeDM_P?usp=drive_link) and place them under `models_rl/`.


## Inference (current setup)

Run all configured datasets/seeds with:
```bash
python exp_doob.py
```

This command:
- runs dataset-specific Doob hyperparameters, and sweep over seeds `0..4`;
- parallelizes jobs across all visible GPUs.

You can control worker concurrency per GPU:

```bash
JOBS_PER_GPU=1 python exp_doob.py
```

## Output

Results are written to:
- `results/<dataset>/doob.csv`



