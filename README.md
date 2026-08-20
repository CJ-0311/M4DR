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

