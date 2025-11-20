# V-BUILD: A graph Transformer-based decision framework for selective updating of vector building data

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)	[![PyG](https://img.shields.io/badge/PyG-2.3%2B-3C2179.svg)](https://www.pyg.org/)	[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official PyTorch implementation of the paper **"V-BUILD: A graph Transformer-based decision framework for selective updating of vector building data"**.

The code implements a **Geometric-aware Graph Transformer** designed to make intelligent decisions (Replace vs. Keep) for vector building updates. It operates directly in the vector domain, utilizing a Unified Correspondence Graph (UCG) to handle complex matching relationships (1:1, 1:n, m:n) and an Adaptive Multi-stream Fusion (AMF) module to integrate multi-granular geometric features.

> **NOTE**: This repository is currently under preparation for code release. The complete implementation will be available upon paper publication.

## 🌟 Key Features

The implementation includes the following core components from the paper:

* **Unified Correspondence Graph (UCG) Construction**:
 * Implements `MultiPolygonGraphBuilder` to convert complex building correspondences into a unified graph structure.
 * Utilizes **Minimum Spanning Trees (MST)** and **Virtual Edges** to connect spatially disjoint buildings within a single matching unit.
 * Encodes local topology (contour edges), non-local context (diagonal edges), and spatial relations (virtual edges).
* **Geometric-aware Graph Transformer**:
 * **Dual-Stream Encoder**: A shared-weight encoder that processes T1 and T2 graphs.
 * **Hybrid Operator**: Combines local message passing (**GINEConv**) for topological details and global **Multi-Head Attention (MHA)** for shape context.
* **Adaptive Multi-stream Fusion (AMF)**:
 * A gating mechanism that dynamically weights five distinct feature streams: Node Representation, Node Difference, Edge Representation, Edge Difference, and Global Morphological Difference.

## 📂 Project Structure

```text
.
├── configs/                  # Model configurations
│   └── graph_transformer_config.py  # Main config for V-BUILD
├── models/                   # Core network definitions
│   └── graph_transformer.py  # Implementation of Encoder and AMF
├── data_loader.py            # Data loading, feature extraction, and UCG preparation
├── graph_builder.py          # Graph construction logic (MST, Virtual Edges)
├── feature_extractor.py      # Geometric feature calculation (Hu moments, compactness, etc.)
├── train.py                  # Training and validation loops with Focal Loss
├── main.py                   # Entry point for the pipeline
└── utils.py                  # Utility functions for logging and metrics
```

## 🛠️ Requirements

* Python 3.8+
* PyTorch 1.12+
* PyTorch Geometric (PyG)
* Shapely
* NumPy
* Pandas
* Scikit-learn

Install dependencies via:
```bash
pip install torch torch-geometric shapely numpy pandas scikit-learn
```

## 🚀 Usage

### 1. Data Preparation
Ensure your data is formatted as GeoJSON files for T1 and T2 phases, along with a CSV file defining the matching relationships.
* **GeoJSON**: Standard FeatureCollection format.
* **CSV**: Should contain matching IDs and change types.

### 2. Configuration
Modify `configs/graph_transformer_config.py` to adjust hyperparameters or ablation switches:
```python
'use_amf': True,          # Enable Adaptive Multi-stream Fusion
```

### 3. Training
Run the main pipeline to start training the Graph Transformer model:

```bash
python main.py --mode train --model graph_transformer
```

The training script will:
1. Construct Unified Correspondence Graphs for building pairs.
2. Extract multi-granular geometric features.
3. Train the model.
4. Save checkpoints and logs to the `experiments/` directory.

## 📊 Performance

The model outputs detailed metrics including Accuracy, Precision, Recall, and F1-score. It is designed to robustly distinguish between "Replace" (significant morphological change) and "Keep" (insignificant change/noise) decisions, even in complex many-to-many matching scenarios.

## 📑 Citation

If you use this code or ideas in your own work, please cite the corresponding paper. 

```bibtex
@article{VBUILD2025,
title = {V-BUILD: A Graph Transformer-based Decision Framework for Selective Updating of Vector Building Data},
author = {},
journal = {},
year = {2025},
volume = {},
number = {},
pages = {},
}
```



