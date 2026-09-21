# 3D Data Processing – Lab 4: Point Cloud Semantic Segmentation with Sparse Convolutions (MinkUNet)

Solution to the course assignment on **sparse 3D convolutional neural networks**: implementing the core components of **MinkUNet** from scratch with the [`spconv`](https://github.com/traveller59/spconv) library, and training it for point-level semantic segmentation on a reduced version of the **SemanticPOSS** dataset.

Covers:

1. **Sparse tensors & sparse convolutions** — understanding `SubMConv3d`, `SparseConv3d`, `SparseInverseConv3d`.
2. Custom **point-to-voxel quantization** (voxelization) and sparse batching (collation).
3. **Inverse coordinate mapping** — projecting voxel-level predictions back onto the raw point cloud.
4. A custom **Sparse Gated Residual Block**, assembled into a full 3D sparse U-Net.
5. Hyperparameter analysis (voxel size, input features, augmentation) vs. **mIoU**, memory usage and runtime.

Assignment authored by [Daniel Fusaro](https://bender97.github.io/) (3D Data Processing 2025/2026, UniPD); this repository contains the completed implementation.

---

## Author

* [@adelinapopa02](https://github.com/adelinapopa02)

---

## Project overview

| File | Role |
|---|---|
| `3DP2026_Lab_4_MinkUNet.ipynb` | Main notebook: voxelization/collation, `SparseGatedResBlock`, MinkUNet architecture, training and evaluation. |
| `Lab_4_MinkUNet_plots.ipynb` | Reads the training logs and produces the plots used in the report. |
| `Lab4_results_005t.txt`, `Lab4_results_010t.txt`, `Lab4_results_020t.txt` | Training/eval logs for the voxel-size sensitivity study (0.05 m / 0.10 m / 0.20 m, `use_remission=True`, 5 epochs) — Task 6.1. |
| `Lab4_results_010f.txt` | Training/eval log for the geometry-only ablation (`voxel_size=0.10`, `use_remission=False`) — Task 6.2, compared against `Lab4_results_010t.txt`. |
| `Lab4_Report.pdf` | Report: voxel-size sensitivity study, remission-vs-geometry-only ablation, class-wise IoU analysis and discussion. |

---

## Run

Open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adelinapopa02/3DP_lab_4/blob/main/3DP2026_Lab_4_MinkUNet.ipynb)

Requirements (Colab handles most of this automatically):

- Python 3.11, CUDA 12.5, PyTorch (preinstalled on Colab)
- `spconv-cu121` (installed by the notebook's setup cell)

Notes:

- Use a **GPU runtime** (`Runtime → Change runtime type → GPU`), and pin the runtime version to `2025.07` to keep Python 3.11.
- Do **not** create virtual environments or reinstall PyTorch/CUDA.
- **Restart the runtime** after installing `spconv-cu121`, before running the cell that imports it.
- The **SemanticPOSS** dataset is provided as a shared Google Drive folder (see the notebook for the link); mount it with a personal `@gmail.com` account — a `@studenti.unipd.it` account will fail to mount the shared folder.
