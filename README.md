# M4DR Framework Overview

M4DR (Multi-Modal Four-Tower Drug Response) is a multimodal deep transfer
learning framework designed to predict drug responses at single-cell
resolution.

The framework addresses the challenge of limited labeled single-cell
drug-response data by transferring knowledge learned from large-scale
bulk pharmacogenomic datasets to single-cell RNA-seq datasets.

M4DR adopts a source-target domain adaptation strategy:

- Source domain:
  labeled bulk RNA-seq cell-line drug response data

- Target domain:
  unlabeled single-cell RNA-seq data

The model integrates four complementary biological and chemical modalities:

1. Gene expression features
2. Pathway activity features
3. Molecular graph representations
4. Molecular fingerprint representations

These heterogeneous features are encoded through a four-tower architecture
and integrated using a cross-modal attention module for drug response
prediction.
# Model Architecture

M4DR consists of four parallel encoding towers:

## 1. Cellular representation learning

### Gene expression encoder

The gene expression tower uses a Transformer-based encoder to learn
high-dimensional transcriptomic representations.

Input:
- normalized gene expression profiles

Output:
- latent cellular embedding


### Pathway activity encoder

The pathway activity tower uses a residual multilayer perceptron
(Residual MLP) to capture nonlinear relationships among biological pathways.

Input:
- pathway activity scores

Output:
- pathway-level biological embedding


The gene expression and pathway embeddings are concatenated to construct
the cellular representation.
## 2. Drug representation learning

### Molecular graph encoder

The molecular graph tower uses a Graph Transformer architecture to learn
structural information from drug molecular graphs.

Input:
- molecular graph representation

Output:
- graph-based drug embedding


### Molecular fingerprint encoder

The fingerprint tower uses a deep residual network to encode Morgan
fingerprint features.

Input:
- Morgan fingerprint vectors

Output:
- fingerprint-based drug embedding


The graph and fingerprint embeddings are fused to generate the final
drug representation.
# Cross-modal Attention Fusion

To model interactions between cellular states and drug properties, M4DR
uses a bidirectional cross-attention module.

The cross-attention mechanism allows:

- cellular features to attend to drug features
- drug features to attend to cellular features

The enhanced cellular and drug representations are concatenated and passed
through an MLP classifier to estimate drug response probabilities.
# Training Strategy

M4DR is trained using a two-stage optimization strategy.

## Stage 1: Source-domain supervised pretraining

In the first stage, M4DR is trained using labeled bulk RNA-seq drug response
data.

The model learns:

- cellular representations from bulk transcriptomic profiles
- drug representations from molecular features
- cell-drug interactions through cross-attention

The source-domain encoders, drug encoders, and prediction head are optimized
using supervised drug response labels.
## Stage 2: Adversarial domain adaptation

After source-domain pretraining, M4DR transfers knowledge from bulk RNA-seq
to single-cell RNA-seq through ADDA-based adversarial domain adaptation.

During adaptation:

- source-domain cellular encoders are frozen
- pretrained weights initialize target-domain encoders
- target-domain gene and pathway encoders are optimized
- domain discriminators align bulk and single-cell feature distributions


Two domain discriminators are introduced:

1. Gene-level discriminator
2. Pathway-level discriminator


The adversarial optimization reduces the distribution discrepancy between
bulk and single-cell feature spaces without requiring target-domain response
labels.
# Usage

## Step 1. Prepare data

Prepare the following input files:

### Source-domain data

Required:

- bulk RNA-seq expression profiles
- pathway activity features
- drug molecular graph representations
- Morgan fingerprints
- drug response labels


### Target-domain data

Required:

- single-cell RNA-seq expression profiles
- pathway activity features


The recommended directory structure:

