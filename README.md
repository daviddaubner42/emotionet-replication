# EmotioNet on DEAP: EEG emotion recognition with a 3-D CNN

A PyTorch re-implementation of **EmotioNet** (Wang, Huang, McCane & Neo, IJCNN 2018), evaluated on the **DEAP** dataset (Koelstra et al., IEEE Trans. Affective Computing, 2012). The model classifies high vs. low **valence** and **arousal** from 32-channel EEG, using a within-subject protocol.

Everything lives in one notebook: [`emotionet_replication.ipynb`](emotionet_replication.ipynb).

## Results

Within-subject, leave-5-trials-out cross-validation (8 folds per subject, 32 subjects). Scores are **balanced accuracy** on trial-level predictions, averaged over folds and then over subjects.

| Target  | Mean balanced accuracy | Chance |
| ------- | ---------------------- | ------ |
| Valence | 0.678                  | 0.5    |
| Arousal | 0.718                  | 0.5    |

Performance varies a lot between people. Some subjects exceed 0.9, while a handful fall at or below chance for one or both targets. The notebook's final section discusses this variability. The per-subject bar chart is saved to `plots/emotionet_within_subject_balanced_accuracy.png` when you run the notebook.

## What the notebook does

1. **Loads DEAP** (preprocessed Python release): 32 subjects × 40 one-minute music-video trials × 32 EEG channels at 128 Hz. Ratings on the 1-9 scale are binarised at 5. Classes are imbalanced (about 21% "high" valence and 23% "high" arousal), so training uses class-weighted cross-entropy and evaluation uses balanced accuracy.
2. **Preprocesses** each trial: drops the 3 s pre-stimulus baseline and arranges the channels on a 7 × 9 scalp grid (empty cells are zero), so the CNN can use spatial kernels.
3. **Builds EmotioNet**: two 3-D convolutions, a spatial-fusion layer that collapses the grid, two temporal 1-D convolutions, max-pooling, and a dense-prediction head that averages per-time-step scores (about 57k parameters).
4. **Cuts trials into windows**: 4 s windows (512 samples) with a 1 s stride, giving 57 windows per trial. All windows of a trial stay on the same side of a split, so there is no leakage between training and validation.
5. **Trains** with Adam (lr 1e-3), up to 30 epochs, early stopping on validation loss (patience 5), and keeps the best-validation-loss weights.
6. **Scores per trial** by averaging the raw outputs of a trial's windows and taking the arg-max.
7. **Runs the within-subject experiment**: 8 folds × 32 subjects = 256 fits per target, then plots per-subject balanced accuracy.

## Getting started

### 1. Install dependencies

Python 3.12 was used. A CUDA GPU is strongly recommended.

```bash
pip install -r requirements.txt
```

### 2. Get the data

DEAP is not included in this repo. Request access from the dataset authors (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and download the **preprocessed Python** version (`data_preprocessed_python`). Place the files so the repo looks like this:

```
.
├── emotionet_replication.ipynb
└── deap-dataset/
    └── data_preprocessed_python/
        ├── s01.dat
        ├── ...
        └── s32.dat
```

The paths are set at the top of the notebook (`DEAP_DIR`, `RESULTS_DIR`, `PLOTS_DIR`, `CACHE_DIR`) and can be changed there.

### 3. Run the notebook

```bash
jupyter notebook emotionet_replication.ipynb
```

Run the cells in order. The notebook creates these folders:

| Folder           | Contents                                                       |
| ---------------- | -------------------------------------------------------------- |
| `preprocessing/` | Cached grid-shaped EEG (`deap_topo_grid_float32.npy`)           |
| `results/`       | Best weights (`.pt`) and training history (`.pkl`) for each fit |
| `plots/`         | Result figures                                                 |

## Configuration

The main settings are in the first code cell.

| Setting                   | Default      | Meaning                                           |
| ------------------------- | ------------ | ------------------------------------------------- |
| `TARGETS`                 | valence, arousal | Affective dimensions to classify              |
| `SUBJECTS`                | `None`       | `None` = all 32 subjects, or a list of indices    |
| `WINDOW_SIZE` / `STRIDE`  | 512 / 128    | Window length and step in samples (4 s / 1 s)     |
| `DROPOUT`                 | 0.5          | Dropout in layers 2, 4 and 5                      |
| `LR`                      | 1e-3         | Adam learning rate                                |
| `N_EPOCHS` / `PATIENCE`   | 30 / 5       | Max epochs and early-stopping patience            |
| `WITHIN_K`                | 5            | Trials held out per fold (40 trials → 8 folds)    |
| `SEED`                    | 26           | Global random seed                                |

## References

- S. Koelstra et al., "DEAP: A Database for Emotion Analysis Using Physiological Signals," *IEEE Transactions on Affective Computing*, 3(1), 2012.
- Y. Wang, Z. Huang, B. McCane and P. Neo, "EmotioNet: A 3-D Convolutional Neural Network for EEG-based Emotion Recognition," *International Joint Conference on Neural Networks (IJCNN)*, 2018.