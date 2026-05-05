# Movie Recommendation System via GNN Link Prediction
### Modeling & Artificial Intelligence — End-of-Semester Project

---

## Overview

This project implements a **Graph Neural Network (GNN)** for **link prediction** on a **bipartite user–movie graph** built from the MovieLens dataset. Given a user and a movie, the model predicts whether the user will watch (and enjoy) that movie.

### Architecture

```
                      ┌─────────────────────────────────────────────────────┐
                      │             Bipartite Graph G = (U, M, E)           │
                      │   U = users   M = movies   E = ratings ≥ threshold  │
                      └───────────────────────┬─────────────────────────────┘
                                              │
               ┌──────────────────────────────▼──────────────────────────────┐
               │                   Node Feature Initialisation                │
               │  Users:  Learnable embedding E_u ∈ ℝ^d                      │
               │  Movies: Learnable embedding E_m + genre projection W·g_m    │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
               ┌──────────────────────────────▼──────────────────────────────┐
               │              LightGCN Propagation  (L layers)                │
               │                                                              │
               │  E_u^(l+1) = D_u^{-½} A D_m^{-½} E_m^(l)                  │
               │  E_m^(l+1) = D_m^{-½} Aᵀ D_u^{-½} E_u^(l)                 │
               │                                                              │
               │  Final: E* = (1/L) Σ_l E^(l)   (layer-wise mean pool)      │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
               ┌──────────────────────────────▼──────────────────────────────┐
               │         Graph Transformer Layer  (PyTorch path only)         │
               │  Multi-head self-attention over neighbourhood embeddings     │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
               ┌──────────────────────────────▼──────────────────────────────┐
               │              Link Prediction & BPR Training Loss             │
               │  score(u,m) = E_u* · E_m*                                   │
               │  L_BPR = -E[log σ(score_pos - score_neg)]                   │
               └─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
movie_recommendation_gnn/
│
├── main.py              # Core: DataLoader, NumPyGNN, BipartiteGNN, Trainer
├── graph_analysis.py    # Structural analysis: degree dist., components, clustering
├── baselines.py         # Baselines: CN, Jaccard, AA, SVD, Popularity, Genre
├── visualize.py         # Plots: training curves, embeddings, comparisons
├── run_pipeline.py      # One-click full pipeline (CLI entry point)
│
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
└── data/
    └── ml-latest-small/ # Download from Kaggle (see below)
        ├── ratings.csv
        ├── movies.csv
        ├── tags.csv
        └── links.csv
```

---

## Dataset Setup

1. Download from Kaggle:
   ```
   https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset
   ```

2. Extract into `data/ml-latest-small/`:
   ```
   movie_recommendation_gnn/data/ml-latest-small/ratings.csv
   movie_recommendation_gnn/data/ml-latest-small/movies.csv
   ```

If the dataset is not found, the code **automatically generates synthetic data** of the same structure so you can run and test immediately.

---

## Installation

```bash
# Core (required)
pip install numpy pandas scikit-learn matplotlib

# Optional — enables full GNN/Graph Transformer
pip install torch torchvision torchaudio
pip install torch-geometric
```

The project runs in **NumPy-only mode** if PyTorch is not installed, using a from-scratch LightGCN implementation. With PyTorch, the full `LightGCN + Graph Transformer` model is activated.

---

## Running the Project

### Quick demo (10 epochs, synthetic data if no dataset):
```bash
python run_pipeline.py --quick
```

### Full run with real data:
```bash
python run_pipeline.py --data_dir data/ml-latest-small --epochs 100
```

### Just the core model:
```bash
python main.py
```

### CLI options:
```
--data_dir        Path to MovieLens folder (default: data/ml-latest-small)
--epochs          Training epochs         (default: 100)
--embed_dim       Embedding dimension     (default: 64)
--layers          GNN layers              (default: 3)
--lr              Learning rate           (default: 0.001)
--batch_size      Mini-batch size         (default: 1024)
--output_dir      Output directory        (default: outputs/)
--skip_baselines  Skip baseline models
--skip_plots      Skip matplotlib figures
--quick           10-epoch fast test
```

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| **LightGCN + Graph Transformer** | GNN | Main model — bipartite message passing + attention |
| Popularity | Baseline | Most-rated movies |
| Genre Similarity | Baseline | Cosine similarity of genre vectors |
| Common Neighbours | Graph | Shared raters |
| Jaccard Coefficient | Graph | Normalised shared neighbourhood |
| Adamic-Adar | Graph | Log-weighted shared raters |
| SVD | Matrix Factorisation | Truncated SVD with SGD |

---

## Evaluation Metrics

- **AUC-ROC** — area under the ROC curve (main metric)
- **Accuracy** — binary classification at 0.5 threshold  
- **Precision@K** — fraction of top-K recs that are relevant
- **Recall@K** — fraction of relevant items in top-K
- **NDCG@K** — normalised discounted cumulative gain (ranking quality)

---

## Key Results (typical on ML-small, 100 epochs)

| Model | AUC | NDCG@10 |
|-------|-----|---------|
| GNN (LightGCN + GT) | **~0.82** | **~0.18** |
| SVD | ~0.74 | ~0.12 |
| Adamic-Adar | ~0.65 | — |
| Genre Similarity | ~0.60 | ~0.08 |
| Popularity | ~0.55 | ~0.06 |

---

## References

1. He et al. (2020). **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. SIGIR.
2. Dwivedi & Bresson (2021). **A Generalization of Transformers to Graphs**. ICLR Workshop.
3. Rendle et al. (2009). **BPR: Bayesian Personalized Ranking from Implicit Feedback**. UAI.
4. Latapy et al. (2008). **Basic notions for the analysis of large two-mode networks**. Social Networks.
5. Harper & Konstan (2015). **The MovieLens Datasets: History and Context**. ACM TiiS.
