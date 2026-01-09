# InfGCN

[InfGCN: Equivariant Neural Operator Learning with Graphon Convolution](https://arxiv.org/abs/2311.10908)

## Abstract

We propose a general architecture that combines the coefficient learning scheme with a residual operator layer for learning mappings between continuous functions in the 3D Euclidean space. Our proposed model is guaranteed to achieve SE(3)-equivariance by design. From the graph spectrum view, our method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, our model can effectively capture the geometric information while preserving equivariance. Through extensive experiments on large-scale electron density datasets, we observed that our model significantly outperformed the current state-of-the-art architectures. Multiple ablation studies were also carried out to demonstrate the effectiveness of the proposed architecture.

![InfGCN Overview](../../docs/infgcn.png)

## Datasets

InfGCN is benchmarked on volumetric electron-density regression. Each example pairs an atomic graph (species, Cartesian coordinates, optional lattice) with charge-density values sampled on a regular 3D grid; collators draw random grid points during training to reduce memory pressure while keeping supervision dense. All datasets ship with deterministic split files and an atom dictionary to keep label spaces consistent across tasks.

- **QM9_EC**: Small organic molecules from QM9 with VASP-generated CHGCAR grids compressed as `*.CHGCAR.lz4` under `dataset_ES/data_qm9` (train 123,835 · val 50 · test 10,000). [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_es.tar), [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_data_split.json). Each sample stores the equilibrium 3D geometry, atom types (typically H, C, N, O, F), and a dense electron-density grid in VASP CHGCAR format; the default loader treats systems as non-periodic (`pbc: false`) so the model must learn to extrapolate density in open boundary conditions.
- **MP_EC (cubic)**: Materials Project crystals serialized as compressed JSON density cubes in `dataset_ES/data_cubic` (train 14,421 · val 1,000 · test 1,000). [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar), [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json). Each record contains a relaxed crystal structure (lattice, fractional coordinates, species up to 84 atom types) and a regular electron-density grid stored as `.json.xz`. In the provided configs, periodicity is disabled at the dataset layer and a cutoff graph is built in Cartesian space, which lets InfGCN focus on local density patterns while still being able to incorporate full-cell information through the optional `cell` field.
- **OMol25_EC**: Electron-density cubes derived from the OMol25 molecular collection (MC_5k subset), expected under `/home/liuxuwei01/processed_output` (train 16 · val 2 · test 2). [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_mc_5k.tar), [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_data_split.json). The dataset focuses on medium-sized organic molecules with richer functional groups; densities are precomputed on fixed grids so experiments primarily probe cross-dataset transfer rather than absolute accuracy.
- **MD17_EC**: Electron-density trajectories for MD17 molecules (ethanol default, plus benzene, phenol, resorcinol, etc.) stored in `dataset_ES/data_md`. [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz). Each trajectory contains hundreds to thousands of finite-temperature geometries per molecule; InfGCN is trained on randomly drawn frames, making this benchmark sensitive to how well equivariance is preserved along dynamical distortions. The splits mirror the published MD17 release, enabling direct comparison to energy/force models when density is used as an auxiliary supervision signal.

## Model

InfGCN is an SE(3)-equivariant neural operator that learns continuous mappings from atomic environments to volumetric electron density.

At the graph level, atoms are embedded using a learnable lookup table over element types and connected with edges built by a radius graph (`cutoff` in the YAML). For each edge, InfGCN computes:

- a unit direction vector and its spherical harmonic expansion up to order `num_spherical` using the e3nn `o3.spherical_harmonics` basis, giving equivariant edge features;  
- a distance encoding via Gaussian radial basis functions implemented by `soft_one_hot_linspace`, with `radial_embed_size`, `gauss_start`, and `gauss_end` controlling the number and spacing of kernels.

These ingredients are fed into a stack of `GCNLayer`s. Each layer performs an equivariant tensor product between node features and edge spherical harmonics, with coefficients predicted by a small MLP `FullyConnectedNet` on the radial embedding. Depending on the `is_fc` flag, the tensor product is either fully connected (`o3.FullyConnectedTensorProduct`) or sparsified (`o3.TensorProduct` with hand-crafted instructions), and an optional self-connection `o3.Linear` is added as a skip path. Nonlinearities are applied either on scalar channels only (`ScalarActivation`, suitable for readouts) or on the norms of higher-order features (`NormActivation`), preserving equivariance.

The network’s operator view is realized in two coupled graphs:

- an **atom–atom graph**, where stacked InfGCN layers propagate information over local neighborhoods, approximating convolution on a dense “graphon” limit via shared tensor-product kernels;  
- a **grid–atom graph**, where each sampled grid point connects to nearby atoms within `grid_cutoff` and a separate `GCNLayer` (`self.residue`) predicts a residual correction in scalar density space.

To evaluate density, a Gaussian atomic orbital expansion (`GaussianOrbital`) is applied at every grid location relative to each atom, and the resulting basis values are contracted with the final equivariant node features. This produces a base density field that is then refined by the residual grid–atom message-passing branch. Periodic boundary conditions, when enabled (`pbc: true` and a `cell` tensor present in the batch), are handled by wrapping vectors into the unit cell via `pbc_vec`, ensuring that both tensor products and density evaluation remain consistent under lattice translations.

Training minimizes mean-squared error between predicted and reference density values on a subsampled set of grid points, optionally masked (`density_mask`) to ignore padding. By combining equivariant tensor products, graphon-inspired message passing, and local orbital expansions, InfGCN captures both sharp bonding features and long-range smooth variations in the electron density with a computational cost that scales as $O(nk)$, where $n$ is the number of atoms and $k$ the average neighbor count.

## Results

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Density MAE</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">infgcn_qm9</td>
            <td nowrap="nowrap">QM9_EC</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_qm9.yaml">infgcn_qm9</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_cubic</td>
            <td nowrap="nowrap">MP_EC (cubic)</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_cubic.yaml">infgcn_cubic</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_omol25</td>
            <td nowrap="nowrap">OMol25_EC</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_omol25.yaml">infgcn_omol25</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md</td>
            <td nowrap="nowrap">MD17_EC (ethanol)</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md.yaml">infgcn_md</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
    </tbody>
</table>

**Note**: Benchmarks are being regenerated in Paddle; metrics and downloadable checkpoints will be published once validation completes. Pretrained QM9 weights: [infgcn_qm9](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_qm9.pdparams)

### Training

```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
# single-gpu training
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
```

### Validation

```bash
# Adjust runtime options via CLI without editing the YAML, e.g. enabling eval-only runs with a saved checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path (*.pdparams)'
```

### Testing

```bash
# Evaluate on the test split using a pretrained checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path (*.pdparams)'
```

### Prediction

```bash
# Run inference with the standalone predictor (uses the dataset paths from the YAML; override via flags if needed).
python electronic_structure/predict.py \
  --config electronic_structure/configs/infgcn/infgcn_qm9.yaml \
  --checkpoint output/infgcn_qm9_best/infgcn_qm9.pdparams \
  --split validation \
  --index 0 \
  --grid_batch_size 20000 \
  --output_dir output/infgcn_qm9_best/vis_val0
# Notes: create a symlink to your data root if it lives elsewhere, e.g. ln -s /path/to/dataset_ES dataset_ES.
# If kaleido is missing, the script writes interactive .html files instead of .png; install kaleido to export PNGs.
```

## Citation
```
@article{deng2023chgnet,
  title={Equivariant neural operator learning with graphon convolution},
  author={Chaoran Cheng and Jian Peng},
  booktitle={Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, December 10-16, 2023},
  month={December},
  year={2023},
}
```
