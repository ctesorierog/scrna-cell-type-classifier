# Single-cell RNA cell-type classifier: underfitting, overfitting and regularization

A dense neural network that classifies single cells into **9 immune cell types** from their gene-expression profiles. The study builds models that underfit, fit well and overfit, then tests which regularization fixes the overfitting.

**Result: 97.1 % test accuracy** (macro F1 0.94) on 7,378 held-out cells, using L2 regularization (λ = 1e-2).

## Data

- Single-cell RNA-seq profiles from two sequencing batches: train has 8,098 cells, test has 7,378.
- Only the 1,199 genes present in both batches are used, so one network works for both.
- 9 classes: B cell, CD4 T, CD8 T, NK, CD14 and FCGR3A monocytes, plasmacytoid dendritic cell, megakaryocyte and hematopoietic stem cell. The classes are very imbalanced: from 2,076 cells down to 17.

The dataset was provided by the course and is not redistributed here (`train.pkl` / `test.pkl`).

## What the notebook covers

1. **Three unregularized models**

   | Model | Size | Train acc. | Val. acc. | Behaviour |
   |---|---|---|---|---|
   | Underfit | 2 × 2 neurons, 1.2k params | 88 % | 85 % | high loss on both sets |
   | Good fit | 19.6k params | 99.98 % | 98.0 % | val loss lowest near epoch 16 |
   | Overfit | 2 × 512 neurons, 882k params | 100 % | 97.9 % | val loss lowest at epoch 2–3, then rises |

2. **Regularization study on the overfit architecture**: L2 (1e-4, 1e-3, 1e-2), L1 (1e-5, 1e-4, 1e-3), L1+L2 and dropout (0.2, 0.3, 0.5).
   - Models are compared on validation cross-entropy tracked separately, because Keras adds the penalty to the reported loss.
   - **L2 at 1e-2** was the most stable: validation error rises only 0.002 after its minimum, against about 0.02 without regularization.
   - Dropout 0.5 reached the lowest single validation error but needs early stopping.
3. **Test evaluation**: classification report and confusion matrix for the selected model.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| B cell | 0.990 | 0.994 | 0.992 |
| CD4 T cell | 0.979 | 0.967 | 0.973 |
| Monocyte CD14 | 0.979 | 0.982 | 0.980 |
| NK cell | 0.983 | 0.993 | 0.988 |
| CD8 T cell | 0.933 | 0.960 | 0.946 |
| Plasmacytoid DC | 0.750 | 1.000 | 0.857 |
| **Overall accuracy** | | | **0.971** |

## Run it

Open `scrna_cell_type_classifier.ipynb` in Google Colab and point `train_path` / `test_path` to the two `.pkl` files. All outputs (training curves, tables and the confusion matrix) are saved in the notebook.

## Stack

Python, Keras / TensorFlow, scikit-learn, pandas, matplotlib, Google Colab.

## Context

Homework 2 (Shallow NNs and Regularization) for the *Deep Learning* course, AI Engineering, Tecnun – Universidad de Navarra, October 2026.
