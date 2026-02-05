# NYC Flood Cascade GNN

## Project Overview

This project represents a pipeline to model how flooding can cause cascading failures in power infrastructure using real-world data and Graph Neural Networks (GNNs).

The system integrates:

- NYC Flood Vulnerability GIS Data
- NYC Power Infrastructure (OpenStreetMap)
- Graph-Based Infrastructure Network Modeling
- Flood Probability Modeling
- Graph Neural Network Failure Prediction

## Problem Statement

Flooding can disrupt critical infrastructure like substations. Because power networks are interconnected, a single flooded substation can cause cascading failures across the grid.

This project predicts: Which substations are most likely to fail during a flood event?

## Methodology

```
    NYC Flood GIS Data
            ↓
    Flood Probability Map
            ↓
    Real Substation Locations (OSM)
            ↓
    Spatial Flood Exposure Mapping
            ↓
    Infrastructure Graph Construction
            ↓
    Cascade Failure Simulation
            ↓
    Graph Neural Network Training
            ↓
    Failure Risk Prediction
```

## Visualizations

### Flood Probability Map

Shows spatial flood risk across NYC using normalized Flood Vulnerability
Index.

### Infrastructure Network Risk Graph

Nodes colored by predicted failure probability.

### Failure Probability Distribution

Histogram showing how many substations fall into high-risk vs low-risk
categories.

## Data Sources

### NYC Flood Data

NYC Flood Vulnerability Index (Storm Surge + Tidal Flood Risk)

### Infrastructure Data

OpenStreetMap Power Infrastructure: - Substations - Plants -
Transformers

## Model Architecture

Graph Neural Network: - Input Features: - Flood Probability - Synthetic Load Proxy - Layers: - GCNConv Layer 1 - ReLU Activation - GCNConv Layer 2 - Output: - Failure Probability Per Node

## Results Interpretation

Each substation receives:

    [ Probability Safe , Probability Fail ]

Example:

    [0.82 , 0.18] → Low Risk
    [0.25 , 0.75] → High Risk

## Tech Stack

- Python
- GeoPandas
- NetworkX
- PyTorch Geometric
- OSMnx
- Matplotlib

## Future Improvements

- Real transmission line topology graph
- Time-step cascade prediction
- Climate scenario comparison (2050 / 2080 flood risk)
