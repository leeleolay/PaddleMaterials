# DiffNMR

[DiffNMR: Diffusion Models for Nuclear Magnetic Resonance Spectra Elucidation](https://doi.org/10.48550/arXiv.2507.08854)

**Qingsong Yang**<sup>#</sup>, **Binglan Wu**<sup>#</sup>, \*\*Xuwei Liu</strong>, Bo Chen, Wei Li, Gen Long<sup>†</sup>, Xin Chen<sup>†</sup>, Mingjun Xiao<sup>†</sup>

<sup>1</sup> Department of Computer Science, University of Science and Technology of China (USTC)

<sup>2</sup> Suzhou Laboratory

<sup>3</sup> Baidu Inc.

<sup>#</sup> Equal contribution   ·   <sup>†</sup> Corresponding authors

---

## Abstract

DiffNMR is an end-to-end framework that infers **molecular structures directly from 1H/13C NMR spectra** using a **conditional discrete diffusion model**. It holistically refines molecular graphs through denoising steps (instead of autoregressive token-by-token generation), improving global consistency and reducing error accumulation. The system couples a **two-stage pretraining** pipeline—(i) **Diffusion Autoencoder (Diff-AE)** for molecular representation + graph decoder pretraining and (ii) **contrastive alignment** between spectra and molecular representations—with a domain-tailored **NMR encoder** that leverages **RBF encodings** for chemical shifts and coupling constants. Inference further benefits from **similarity-based filtering** and optional **retrieval-initialized sampling**, yielding strong Top‑k accuracy and high Tanimoto similarity across molecule sizes.

> Paper: arXiv:2507.08854 (v1). DOI: 10.48550/arXiv.2507.08854.

---

## Highlights

* **End-to-end spectrum→structure** on 1H & 13C NMR.
* **Discrete graph diffusion** (denoise molecular graphs instead of autoregressive SMILES).
* **NMR encoder** with **RBF** embeddings for shifts (and J-coupling) + learnable embeddings for multiplicity & integrals; **bi-directional cross‑attention** to fuse 1H/13C information.
* **Two-stage pretraining**: Diff‑AE (molecular encoder + graph diffusion decoder) → **contrastive learning** to align NMR & molecular spaces.
* **Inference enhancements**: **similarity filtering** (cosine scoring in latent space) and **retrieval‑initialized sampling**.

---

## Dataset & Metrics

* **MSD multimodal spectroscopic dataset**: \~7.9e5 molecules (from USPTO), each with simulated **1H NMR**, **13C NMR**, **HSQC**, **IR**, **MS**; molecules span **5–35 heavy atoms**.
* **Evaluation**: **Top‑k accuracy** (exact‑match structure among top candidates) and **Tanimoto similarity** (Morgan fp, radius 2, 2048 bits).

> Observations
>
> * **1H+13C** outperforms single‑modality inputs.
> * Providing **molecular formula** improves accuracy across sizes.
> * **Similarity filtering** and **retrieval initialization** substantially boost Top‑1 accuracy and average Tanimoto, especially for larger molecules.

---

## Method at a Glance

**Architecture**

* **Molecular encoder** (graph transformer) → compact molecular representation.
* **Graph diffusion decoder** (discrete denoising on nodes/edges; FiLM‑conditioned by NMR embedding).
* **NMR encoder**: RBF(δ) for shifts; embeddings for multiplicity/integrals; RBF(J) for couplings; transformers per modality; **bi‑directional cross‑attention** (1H↔13C); pooled & fused to a conditioning vector.

**Training**

1. **Stage‑1: Diff‑AE** — pretrain molecular encoder + diffusion decoder via reconstruction.
2. **Stage‑2: Contrastive** — freeze molecular encoder; train NMR encoder to align to the molecular space with **InfoNCE**.
3. **Fine‑tuning** — end‑to‑end spectra→graph with diffusion decoder conditioned on NMR embedding.

**Inference**

* Optional **retrieval‑init** (start from nearest neighbor in latent space) vs **random‑init**.
* **Similarity filtering**: rank sampled candidates by cosine similarity between predicted molecule & input NMR embeddings.

---

## Getting Started

### 0) Environment

* **Framework**: \[PaddlePaddle ≥ 2.6] and **PaddleMaterial** toolkit.
* Install PaddleMaterial (example):

```bash
# from source
git clone https://github.com/PaddlePaddle/PaddleMaterial.git
cd PaddleMaterial
pip install -r requirements.txt
pip install -e .
```

> Note: CUDA/NCCL versions should match your device drivers. See Paddle/PaddleMaterial docs for details.

### 1) Prepare NMR inputs

Create a JSON/CSV containing 1H & 13C peaks per sample. Example JSON (one sample):

```json
{
  "id": "sample_0001",
  "formula": "C10H12O2",  // optional
  "h1": [
    {"shift": 3.62, "multiplicity": "t", "integral": 2, "j": [1.28]},
    {"shift": 5.81, "multiplicity": "s", "integral": 1},
    {"shift": 4.62, "multiplicity": "d", "integral": 2, "j": [6.78]}
  ],
  "c13": [173.0, 62.3, 53.8]
}
```

* **1H**: shift (ppm), multiplicity {s,d,t,q,m,...}, integral (≈ #H), optional **J** list (Hz)
* **13C**: shift (ppm)

### 2) Run inference (spectrum → structure)

Example commands (paths may differ by repo layout):

```bash
# Option A: by predefined model name (weights auto‑loaded if provided by PaddleMaterial)
python spectrum_elucidation/sample.py \
  --model_name diffnmr \
  --mode by_dataloader \
  --data_path ./data/nmr_test.json \
  --save_path ./results_diffnmr

# Option B: by explicit config/weights
python spectrum_elucidation/sample.py \
  --config_path ./configs/diffnmr/sample.yaml \
  --checkpoint_path ./checkpoints/diffnmr/best.pdparams \
  --mode by_dataloader \
  --data_path ./data/nmr_test.json \
  --save_path ./results_diffnmr
```

Useful config keys (YAML):

```yaml
sampling:
  num_samples: 64        # candidates per spectrum
  steps: 500             # diffusion steps (random init)
retrieval:
  enabled: true          # retrieval‑initialized sampling
  topk: 32               # pool size from DB
  index_path: ./db/index.faiss
filtering:
  enabled: true          # similarity filtering
  keep_topk: 10
```

### 3) Training (optional)

```bash
# Stage‑1 Diff‑AE pretraining (molecular reconstruction)
python spectrum_elucidation/train.py --config_path ./configs/diffnmr/pretrain_diffae.yaml

# Stage‑2 contrastive pretraining (NMR↔molecule alignment)
python spectrum_elucidation/train.py --config_path ./configs/diffnmr/pretrain_contrast.yaml

# End‑to‑end fine‑tuning
python spectrum_elucidation/train.py --config_path ./configs/diffnmr/finetune.yaml
```

---

## Results (summary)

* **1H+13C** achieves the best Top‑1/Top‑k; adding **molecular formula** improves all sizes (≤15/≤20/≤25 HAC).
* **Similarity filtering** and **retrieval initialization** notably improve **Top‑1** and **avg. Tanimoto**, with the largest gains on higher HAC.

> See the paper for detailed tables/figures and ablations (RBF encoding vs discrete, pretraining stages, etc.).

---

## Repository Layout (suggested)

```
PaddleMaterials/
  spectrum_elucidation/
    configs/DiffNMR/
      DiffNMR_DiffGraphFormer.yaml
      DiffNMR_NMRNet.yaml
      DiffNMR.yaml
    train.py
    sample.py
```

---

## Citation

If you use DiffNMR, please cite:

```bibtex
@misc{yang2025diffnmr,
  title         = {DiffNMR: Diffusion Models for Nuclear Magnetic Resonance Spectra Elucidation},
  author        = {Yang, Qingsong and Wu, Binglan and Liu, Xuwei and Chen, Bo and Li, Wei and Long, Gen and Chen, Xin and Xiao, Mingjun},
  year          = {2025},
  eprint        = {2507.08854},
  archivePrefix = {arXiv},
  primaryClass  = {physics.chem-ph},
  doi           = {10.48550/arXiv.2507.08854},
  url           = {https://arxiv.org/abs/2507.08854}
}
```

---

## License

This repository is released under the Apache-2.0 license (unless otherwise stated). See `LICENSE` for details.

## Acknowledgements

Supported by the National Science and Technology Major Project (2023ZD0120702) and Basic Research Program of Jiangsu (BK20231215). We thank contributors of PaddlePaddle & PaddleMaterials

---

