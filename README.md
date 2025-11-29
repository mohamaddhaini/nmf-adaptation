# Domain Adaptation in Regression by Aligning Non-negative Decompositions
# Adaptation de domaine en régression par alignement de décompositions non-négatives

This repository contains the reference implementation for the GRETSI 2022 paper on
domain adaptation for regression by aligning non-negative matrix decompositions.
The codebase demonstrates how to constrain the latent feature space of a ResNet-18
regressor with an additional NMF-based alignment regularizer so that predictions
on a source domain transfer more reliably to an unseen target domain.

## Repository Structure
- `dSprites/model.py`: ResNet/AlexNet backbones stripped from their classifiers.
- `dSprites/transform.py`: Torchvision transformation utilities for train/test.
- `dSprites/read_data.py`: Minimal dataset and batching helpers for regression.
- `dSprites/train_nmf.py`: Main training loop with NMF regularization term.
- `dSprites/*.txt`: File lists containing the image paths and labels per domain.

## Requirements
- Python 3.8+ with `torch`, `torchvision`, `numpy`, `scikit-learn`, and `matplotlib`.
- GPU with CUDA is recommended for the full training regime but CPU works for dry runs.
- Optional: `accimage` for faster JPEG decoding (code automatically falls back to PIL).

Create an environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install torch/torchvision manually
```

## Dataset Preparation
We use the dSprites-based synthetic regression benchmark introduced in the paper.
Download and extract the list files that point to the images and regression labels:

- `color.tgz`: https://cloud.tsinghua.edu.cn/f/649277b5d5de4c0f8fa2/?dl=1
- `noisy.tgz`: https://cloud.tsinghua.edu.cn/f/35cc1489c7b34ee6a449/?dl=1
- `scream.tgz`: https://cloud.tsinghua.edu.cn/f/583ccf6a795448ec9edd/?dl=1

Each archive contains `<split>.txt` files with one path per line followed by the
corresponding regression targets. Update the paths inside `train_nmf.py` to point
to the directory where you extracted the archives as well as the location of this
project (`path_to_project_folder` and `path_to_data` placeholders).

## Training
1. Adjust `batch_size`, `data_transforms`, and file names inside `dSprites/train_nmf.py`
   if you work with different resolutions or dataset splits.
2. Run the training script from the project root:

```bash
cd dSprites
python train_nmf.py --any-additional-flags
```

The script trains a ResNet-18 regressor on the source domain while computing an NMF
alignment loss between the source and target features. The model checkpoint with the
lowest validation MAE is saved under `path_to_save_results`.

## Evaluation
The helper `Regression_test` in `train_nmf.py` evaluates the mean squared and absolute
errors on the held-out test split. You can re-run evaluation by loading a checkpoint
and calling the function with the appropriate dataloaders:

```python
model = Model_Regression().to(device)
model.load_state_dict(torch.load("path_to_save_results/nmf"))
Regression_test(dset_loaders, model.predict_layer)
```

## Extending the Project
- Swap the backbone in `model.py` to experiment with lighter/heavier architectures.
- Modify the `match_nmf` function to include sparsity penalties or other divergences.
- Integrate additional domains by adding their `<name>.txt` files to the loaders.

## Citation
If you find this work useful for your research, please cite:

Dhaini, Mohamad, et al. "Adaptation de domaine en régression par alignement de décompositions non-négatives." 28-ème Colloque GRETSI sur le Traitement du Signal et des Images. 2022.

## Paper
https://gretsi.fr/data/colloque/pdf/2022_dhaini922.pdf
