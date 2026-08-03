# Property Prediction

## 1.Introduction

Property Prediction (PP) targets rapid, first-principles-level estimation of key crystalline properties—formation energy, band gap, elastic moduli, ionic conductivity, and more—without performing new density-functional-theory calculations. The workflow mirrors modern ML interatomic-potential pipelines but shifts the label space from forces to scalar and tensor observables. Starting from crystal structure files (CIF), an automated converter builds atom–bond graphs enriched with chemical descriptors and symmetry-aware positional encodings. Equivariant graph neural networks, or transformer-based variants, are then trained on tens of thousands of reference entries. By collapsing months of high-throughput DFT time into minutes of GPU inference, PP empowers data-driven discovery of semiconductors, catalysts and functional

## 2.Models Matrix

| **Supported Functions**                      | **[MEGNet](./configs/megnet/README.md)** | **[iComformer](./configs/comformer/README.md)** | **[DimeNet++](./configs/dimenet++/README.md)** | **[SphereNet](./configs/spherenet/README.md)** |
| -------------------------------------------- | :--------------------------------------: | :-------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| **Forward Prediction · Materials Properties**|                                          |                                               |                                                |                                                |
| Formation energy                             |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| Band gap                                     |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| Bulk modulus                                 |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| Shear modulus                                |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| Young’s modulus                              |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| Adsorption energy                            |                    🚧                    |                       🚧                      |                       🚧                       |                       —                        |
| **Forward Prediction · Molecular Properties**|                                          |                                               |                                                |                                                |
| $\mu$ (dipole moment)                              |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $\alpha$ (isotropic polarizability)               |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $\varepsilon_{\text{HOMO}}$                        |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $\varepsilon_{\text{LUMO}}$                        |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $\Delta\varepsilon$ (HOMO-LUMO gap)                 |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $\langle R^2 \rangle$ (electronic spatial extent) |                     —                    |                       —                       |                       —                       |                       ✅                       |
| ZPVE (zero-point vibrational energy)                |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $U_0$ (internal energy at 0 K)                      |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $U$ (internal energy at 298.15 K)                   |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $H$ (enthalpy at 298.15 K)                          |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $G$ (free energy at 298.15 K)                       |                     —                    |                       —                       |                       —                       |                       ✅                       |
| $C_v$ (heat capacity)                               |                     —                    |                       —                       |                       —                       |                       ✅                       |
| **ML Capabilities · Training**               |                                          |                                               |                                                |                                                |
| Single-GPU                                   |                    ✅                    |                       ✅                      |                       ✅                       |                       ✅                       |
| Distributed training (eager)                 |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| Mixed precision (AMP)                        |                    —                     |                       —                       |                       —                        |                       —                        |
| Fine-tuning                                  |                    ✅                    |                       ✅                      |                       ✅                       |                       🚧                       |
| Uncertainty / Active Learning                |                    —                     |                       —                       |                       —                        |                       —                        |
| Dynamic→Static graphs                        |                    ✅                    |                       ✅                      |                       ✅                       |                       ✅                       |
| CINN Trainer · single-GPU FP32               |                    ✅                    |                       ✅                      |                       ✅                       |                       ✅                       |
| CINN Trainer · AMP / distributed             |                    —                     |                       —                       |                       —                        |                       —                        |
| **ML Capabilities · Predict**                |                                          |                                               |                                                |                                                |
| Distillation / Pruning                       |                    —                     |                       —                       |                       —                        |                       —                        |
| Standard inference                           |                    ✅                    |                       ✅                      |                       ✅                       |                       ✅                       |
| Distributed inference                        |                    —                     |                       —                       |                       —                        |                       —                        |
| Compiler-level inference · single-GPU FP32   |                    ✅                    |                       ✅                      |                       ✅                       |                       ✅                       |
| **Datasets**                                 |                                          |                                               |                                                |                                                |
| **Materials Project**                        |                                          |                                               |                                                |                                                |
| MP2024                                       |                    ✅                    |                       ✅                      |                       —                        |                       —                        |
| MP2020                                       |                    ✅                    |                       ✅                      |                       —                        |                       —                        |
| MP2018                                       |                    ✅                    |                       ✅                      |                       —                        |                       —                        |
| **JARVIS**                                   |                                          |                                               |                                                |                                                |
| dft_2d                                       |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| dft_3d                                       |                    ✅                    |                       ✅                      |                       —                        |                       —                        |
| **Alexandria**                               |                                          |                                               |                                                |                                                |
| pbe_2d                                       |                    ✅                    |                       ✅                      |                       —                        |                       —                        |
| **ML2DDB🌟**                                 |                    ✅                    |                       ✅                      |                       ✅                       |                       —                        |
| **QM9**                                      |                    —                    |                       —                       |                       ✅                       |                       ✅                       |

**Legend:** ✅ Verified · 🧪 Implemented, pending validation · 🚧 In development · `-` Not supported · 🌟 Original Work

## 3.CINN Trainer and Predictor

`MEGNetPlus`, `iComformer`, `DimeNetPlusPlus`, and `SphereNet` provide the same
opt-in end-to-end CINN path through the normal Trainer and Predictor. This
covers all repository property-prediction configurations: 36 MEGNet, 8
iComformer, 4 DimeNet++, and 12 SphereNet YAML files. Configuration coverage
does not mean that every full training schedule or paper metric has been run.

The validated runtime is PaddlePaddle GPU 3.3.1, one CUDA device, and FP32.
Select it at the workflow level:

```bash
CUDA_VISIBLE_DEVICES=0 python property_prediction/train.py \
  -c property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml \
  Trainer.execution_backend=cinn \
  Trainer.use_amp=false

python property_prediction/predict.py \
  --config_path property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml \
  --checkpoint_path /path/to/checkpoints/best.pdparams \
  --device gpu:0 \
  --cif_file_path property_prediction/example_data/cifs/mp-18767-LiMnO2.cif \
  --save_path output/megnet_cinn_prediction.csv \
  Predict.execution_backend=cinn
```

For SphereNet, use `--xyz_file_path` with a QM9 configuration. All 12 current
SphereNet property YAMLs set `energy_and_force=false` and are in scope;
`energy_and_force=true` and its second-order force-loss backward are not
supported by CINN. AMP, distributed CINN execution, and persistent compiled
runtime export are also unsupported. PaddlePaddle 3.3.1 additionally has a
known eager-backward limitation for DimeNet++ two-node graphs with zero valid
triplets; that edge case has forward coverage but no eager/CINN backward parity
claim.

See the [shared runtime contract](../docs/cinn_end_to_end.md) for lifecycle,
validation levels, registered-model prediction, and all limitations. The
[MEGNet adapter record](../docs/megnet_cinn_phase1.md) retains its detailed
phase-one numerical and workflow evidence.
