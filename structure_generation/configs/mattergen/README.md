# MatterGen

[A generative model for inorganic materials design](https://www.nature.com/articles/s41586-025-08628-5)

## Abstract

The design of functional materials with desired properties is essential in driving technological advances in areas like energy storage, catalysis, and carbon capture. Generative models provide a new paradigm for materials design by directly generating novel materials given desired property constraints, but current methods have low success rate in proposing stable crystals or can only satisfy a limited set of property constraints. Here, we present MatterGen, a model that generates stable, diverse inorganic materials across the periodic table and can further be fine-tuned to steer the generation towards a broad range of property constraints. Compared to prior generative models, structures produced by MatterGen are more than twice as likely to be novel and stable, and more than 10 times closer to the local energy minimum. After fine-tuning, MatterGen successfully generates stable, novel materials with desired chemistry, symmetry, as well as mechanical, electronic and magnetic properties. As a proof of concept, we synthesize one of the generated structures and measure its property value to be within 20 % of our target. We believe that the quality of generated materials and the breadth of MatterGen’s capabilities represent a major advancement towards creating a foundational generative model for materials design.

![MatterGen Overview](../../docs/mattergen.png)

## Datasets

MatterGen is trained and fine-tuned on crystal structure corpora where each sample provides a CIF string (converted to primitive, Niggli-reduced cells during preprocessing) plus optional property columns used for conditioning. CSV manifests are cached into `./data/*_cache/` on first load to accelerate training, and structures are reconstructed via `BuildStructure` and pymatgen before being converted into graphs for the denoiser.

- **MP-20**: 45,231 stable Materials Project structures with at most 20 atoms per conventional cell. Following the upstream release (see `jointContribution/mattergen/data-release/mp-20/README.md`), samples containing Tc, Pm, or elements with $Z \ge 84$ are removed; all structures are re-relaxed with a consistent PBE DFT workflow; and training data are filtered to energy above hull $E_\text{hull} \le 0.1$ eV/atom. The dataset is stored as CSV files with columns such as:

  - `material_id`: Materials Project identifier (e.g. `mp-10009`),  
  - `cif`: CIF string for the relaxed structure,  
  - `formation_energy_per_atom`, `band_gap`, `e_above_hull`, and space-group information,  
  - additional task-specific columns when available.

  The splits used in this repo match the public MatterGen / CDVAE MP-20 benchmark: train 27,136 · val 9,047 · test 9,046. The official MP-20 CSVs for structure generation can be downloaded from [mp_20.zip](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) and referenced by the configs via `./data/mp_20_chemical_system/*.csv`.

- **Alex-MP-20**: A large-scale merge of the Alexandria dataset ([Schmidt et al., 2022](https://archive.materialscloud.org/record/2022.126)) with MP-20, used for pretraining and multi-property fine-tuning. As documented in `jointContribution/mattergen/data-release/alex-mp/README.md`, structures containing Tc, Pm, or elements with $Z \ge 84$ are removed, all structures are re-relaxed with PBE, and for training the subset with more than 20 atoms or $E_\text{hull} > 0.1$ eV/atom is discarded. The resulting Alex-MP-20 table adds multiple property columns (e.g. `dft_band_gap`, `dft_mag_density`, `ml_bulk_modulus`, `space_group`, `hhi_score`, `energy_above_hull`) that serve as conditioning signals.

In this directory, each YAML config corresponds to a specific choice of property columns and conditioning scheme. For example, `mattergen_mp20.yaml` trains an unconditional base model on MP-20, whereas `mattergen_alex_mp20_dft_band_gap.yaml` fine-tunes a base model on Alex-MP-20 with DFT band gap as the conditioning variable, and `mattergen_alex_mp20_dft_mag_density_hhi_score.yaml` enables joint multi-property control.

## Model

MatterGen models crystal generation as a diffusion process over lattice matrices, fractional atomic coordinates, and discrete atom types. In the Paddle implementation (`ppmat.models.mattergen.MatterGen` and `MatterGenWithCondition`), three coupled noise processes are defined:

- a *continuous* diffusion on lattice matrices, with symmetry-preserving Gaussian noise (`make_noise_symmetric_preserve_variance`) and a dedicated scheduler;  
- a *wrapped normal* diffusion on fractional coordinates inside the unit cell, implemented via a wrapped normal score matching loss so that coordinates remain periodic and can be taken modulo 1.0 without discontinuities;  
- a *discrete* diffusion (D3PM-style) on atom types, where categorical noise is injected into zero-based element indices and a hybrid cross-entropy / score-matching objective is used (`d3pm_hybrid_lambda` controls the balance).

At each training step, clean structures from MP-20 or Alex-MP-20 are perturbed by these three processes at a random diffusion time $t$, and a shared SE(3)-equivariant denoiser (`GemNetTDenoiser`) receives the noisy lattice, noisy fractional coordinates, noisy atom types, and a sinusoidal time embedding. The denoiser is a GemNet-T–like message-passing network over periodic graphs, where:

- nodes correspond to atoms with element embeddings;  
- edges connect neighbors according to a cutoff and encode distance / angular information as in GemNet;  
- the time embedding is injected into node and/or edge features so that the network can learn time-dependent scores for each diffusion scale.

The unconditional `MatterGen` learns to reconstruct all three score fields (lattice, coordinates, atom types), with separate loss weights (`lattice_loss_weight`, `coord_loss_weight`, `atom_loss_weight`). The conditional variant `MatterGenWithCondition` augments this with a `SetEmbeddingType` module that encodes one or more property values (e.g. band gap, magnetic density, bulk modulus, chemical system, space group) into a global conditioning vector. Through classifier-free guidance, the model is trained to sometimes drop the conditioning signal (`_USE_UNCONDITIONAL_EMBEDDING`) and at sampling time a guidance scale amplifies the difference between conditional and unconditional scores, enabling strong steering toward the desired property targets.

Sampling starts from pure noise in lattice, coordinates, and atom types and runs a predictor–corrector sampler for a fixed number of steps (typically 1,000). At each time step, the denoiser predicts score fields which are used to:

1. update fractional coordinates via wrapped normal predictor–corrector updates;  
2. update the lattice via an Ornstein–Uhlenbeck–style continuous-time diffusion step;  
3. update atom types via a discrete diffusion step using the learned logits.

For evaluation and generation, the mean predictions from the last step are taken as the final structure (lattice, atom types, fractional coordinates), which can then be exported as CIFs and optionally relaxed by DFT to measure stability, novelty, uniqueness, and property fidelity as in the original MatterGen paper.

## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model Name</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Val(loss)</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">mattergen_mp20</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3721</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20.yaml">mattergen_mp20</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_mp20_chemical_system</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3121</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_chemical_system.yaml">mattergen_mp20_chemical_system</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_chemical_system.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_mp20_dft_band_gap</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3575</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_dft_band_gap.yaml">mattergen_mp20_dft_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_band_gap.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_mp20_dft_bulk_modulus</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.2942</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_dft_bulk_modulus.yaml">mattergen_mp20_dft_bulk_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_bulk_modulus.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_mp20_dft_mag_density</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3620</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_dft_mag_density.yaml">mattergen_mp20_dft_mag_density</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_mag_density.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2960</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20.yaml">mattergen_alex_mp20</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_dft_band_gap</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.3101</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_dft_band_gap.yaml">mattergen_alex_mp20_dft_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_band_gap.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_chemical_system</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2289</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_chemical_system.yaml">mattergen_alex_mp20_chemical_system</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_chemical_system.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_dft_mag_density</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2881</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_dft_mag_density.yaml">mattergen_alex_mp20_dft_mag_density</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_mag_density.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_ml_bulk_modulus</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2811</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_ml_bulk_modulus.yaml">mattergen_alex_mp20_ml_bulk_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_ml_bulk_modulus.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_space_group</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2795</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_space_group.yaml">mattergen_alex_mp20_space_group</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_space_group.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_chemical_system_energy_above_hull</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2272</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_chemical_system_energy_above_hull.yaml">mattergen_alex_mp20_chemical_system_energy_above_hull</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">mattergen_alex_mp20_dft_mag_density_hhi_score</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2803</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20_dft_mag_density_hhi_score.yaml">mattergen_alex_mp20_dft_mag_density_hhi_score</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.zip">checkpoint | log</a></td>
        </tr>
    </body>
</table>

### Training
```bash
# mp20 dataset, without conditional constraints
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml

# mp20 dataset, with chemical system constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml

# mp20 dataset, with dft_band_gap constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml

# mp20 dataset, with dft_bulk_modulus constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml

# mp20 dataset, with dft_mag_density constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml


# alex_mp20 dataset, without conditional constraints
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml

# alex_mp20 dataset, with dft_band_gap constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_band_gap.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_band_gap.yaml

# alex_mp20 dataset, with chemical system constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system.yaml

# alex_mp20 dataset, with dft_mag_density constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density.yaml

# alex_mp20 dataset, with ml_bulk_modulus constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_ml_bulk_modulus.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_ml_bulk_modulus.yaml

# alex_mp20 dataset, with space_group constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_space_group.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_space_group.yaml

# alex_mp20 dataset, with chemical system and energy above hull constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.yaml

# alex_mp20 dataset, with dft_mag_density and hhi_score constraints, pre-trained model is mattergen_alex_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True

# mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with chemical system constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with dft_band_gap constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with dft_bulk_modulus constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with dft_mag_density constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with dft_band_gap constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_band_gap.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with chemical system constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with dft_mag_density constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with ml_bulk_modulus constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_ml_bulk_modulus.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with space_group constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_space_group.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with chemical system and energy above hull constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# alex_mp20 dataset, with dft_mag_density and hhi_score constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

# mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with chemical system constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with dft_band_gap constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with dft_bulk_modulus constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'

# mp20 dataset, with dft_mag_density constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'

# Since the alex_mp20 dataset does not include a test set, we cannot utilize the test command.
```

### Sample
```bash
# This command is used to predict the  crystal structure using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The chemical_formula parameter is used to specify the chemical formula of the crystal structure to be predicted.
# The prediction results will be saved in the folder specified by the `save_path` parameter, with the default set to `result`.

# mp20 dataset, without conditional constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_num_atoms' --num_atoms=4
# or
python structure_generation/sample.py --model_name='mattergen_mp20' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.

python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20.yaml' --checkpoint_path='./output/mattergen_mp20/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_num_atoms' --num_atoms=4
# or
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20.yaml' --checkpoint_path='./output/mattergen_mp20/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_dataloader'


# mp20 dataset, with chemical system constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_chemical_system' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_chemical_system/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml' --checkpoint_path='./outpout/mattergen_mp20_chemical_system/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_chemical_system/' --mode='by_dataloader'

# mp20 dataset, with dft_band_gap constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_dft_band_gap' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_dft_band_gap/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml' --checkpoint_path='./outpout/mattergen_mp20_dft_band_gap/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_dft_band_gap/' --mode='by_dataloader'

# mp20 dataset, with dft_bulk_modulus constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_dft_bulk_modulus' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_dft_bulk_modulus/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml' --checkpoint_path='./outpout/mattergen_mp20_dft_bulk_modulus/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_dft_bulk_modulus/' --mode='by_dataloader'

# mp20 dataset, with dft_mag_density constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_dft_mag_density' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_dft_mag_density/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml' --checkpoint_path='./outpout/mattergen_mp20_dft_mag_density/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_dft_mag_density/' --mode='by_dataloader'

# alex_mp20 dataset, without conditional constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20/' --mode='by_dataloader'

# alex_mp20 dataset, with dft_band_gap constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_dft_band_gap' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_dft_band_gap/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_dft_band_gap.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_dft_band_gap/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_dft_band_gap/' --mode='by_dataloader'

# alex_mp20 dataset, with chemical_system constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_chemical_system' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_chemical_system/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_chemical_system/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_chemical_system/' --mode='by_dataloader'

# alex_mp20 dataset, with dft_mag_density constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_dft_mag_density' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_dft_mag_density/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_dft_mag_density/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_dft_mag_density/' --mode='by_dataloader'

# alex_mp20 dataset, with ml_bulk_modulus constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_ml_bulk_modulus' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_ml_bulk_modulus/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_ml_bulk_modulus.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_ml_bulk_modulus/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_ml_bulk_modulus/' --mode='by_dataloader'

# alex_mp20 dataset, with space_group constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_space_group' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_space_group/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_space_group.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_space_group/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_space_group/' --mode='by_dataloader'

# alex_mp20 dataset, with chemical_system and energy_above_hull constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_chemical_system_energy_above_hull' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_chemical_system_energy_above_hull/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_chemical_system_energy_above_hull/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_chemical_system_energy_above_hull/' --mode

# alex_mp20 dataset, with dft_mag_density and hhi_score constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20_dft_mag_density_hhi_score' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20_dft_mag_density_hhi_score/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20_dft_mag_density_hhi_score/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20_dft_mag_density_hhi
```

## Citation
```
@article{zeni2025generative,
  title={A generative model for inorganic materials design},
  author={Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and others},
  journal={Nature},
  pages={1--3},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
