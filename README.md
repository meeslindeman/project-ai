# Hyperbolic Attention Models for Graph Node Classification

## Introduction

This repository provides a controlled experimental framework for studying hyperbolic attention mechanisms in graph node classification. It compares fully hyperbolic and hybrid hyperbolic–Euclidean attention models against Euclidean baselines, with a focus on isolating the effect of attention geometry while keeping other architectural components minimal.

## Overview

This project implements three model architectures:

* **personal**: Custom hyperbolic attention model with Lorentz geometry
* **hypformer**: Hyperbolic transformer architecture
* **euclidean**: Euclidean baseline model

Supported datasets:

* **Heterophilous**: Chameleon, Squirrel, Actor (Film)
* **Homophilous**: Airport, Disease

## Setup

 If you're using the provided `environment.yml` (recommended): 
 ```bash conda env create -f environment.yml 
 conda activate hyperbolic
 ```

### Train proposed model

```bash
python train.py \
  --dataset chameleon \
  --model personal \
  --split 0 \
  --hidden_dim 64 \
  --num_layers 2 \
  --num_heads 1 \
  --curvature 1.0 \
  --optimizer RiemannianAdam \
  --lr 5e-3
```



### Key Arguments

* `--dataset`: Dataset name (chameleon, squirrel, actor, airport, disease)
* `--model`: Model architecture (personal, hypformer, euclidean)
* `--split`: Split index 0-9
* `--hidden_dim`: Hidden dimension (default: 64)
* `--num_layers`: Number of layers (default: 2)
* `--num_heads`: Number of attention heads (default: 1)
* `--curvature`: Hyperbolic curvature (default: 1.0)
* `--train_curvature`: Make curvature trainable
* `--optimizer`: Optimizer (Adam, AdamW, RiemannianAdam, RiemannianSGD)
* `--lr`: Learning rate (default: 5e-3)
* `--epochs`: Training epochs (default: 500)
* `--patience`: Early stopping patience (default: 100)