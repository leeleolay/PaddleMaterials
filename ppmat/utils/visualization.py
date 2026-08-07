# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from pathlib import Path

import rdkit
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D

from ppmat.utils import logger
from ppmat.utils.ext_rdkit import mol_from_graphs

try:
    import imageio
except ImportError:  # pragma: no cover - optional visualization dependency
    imageio = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional visualization dependency
    go = None

try:
    from IPython.display import Image
    from IPython.display import display
except ImportError:
    Image = None
    display = None


class MolecularVisualization:
    def __init__(self, dataset_infos, output_dir):
        self.dataset_infos = dataset_infos
        self.result_path = os.path.join(output_dir, "graph/")

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """Convert graphs to rdkit molecules using the dataset atom decoder."""

        return mol_from_graphs(
            self.dataset_infos.atom_decoder,
            node_list,
            adjacency_matrix,
            bond_decoder=self.dataset_infos.vocab["bond"],
        )

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int):
        if not os.path.exists(path):
            os.makedirs(path)
        requested = max(0, int(num_molecules_to_visualize))
        if requested > len(molecules):
            logger.info(f"Shortening to {len(molecules)}")
            requested = len(molecules)
        logger.info(f"Visualizing {requested} of {len(molecules)}")
        for i in range(requested):
            file_path = os.path.join(path, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0], molecules[i][1])
            if mol is None:
                logger.warning(f"Skipping invalid molecule at index {i}.")
                continue
            try:
                Draw.MolToFile(mol, file_path)
            except rdkit.Chem.KekulizeException:
                logger.info("Can't kekulize molecule")

    def visualizeNmr(
        self,
        batch_id,
        molecules: list,
        molecules_true,
        num_molecules_to_visualize: int,
    ):
        path = os.path.join(self.result_path, f"batch_{batch_id}_predicted")
        path_true = os.path.join(self.result_path, f"batch_{batch_id}_true")
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_true):
            os.makedirs(path_true)
        available = min(len(molecules), len(molecules_true))
        requested = max(0, int(num_molecules_to_visualize))
        if requested > available:
            logger.info(f"Sampling: Shortening to {available}")
            requested = available
        for i in range(requested):
            file_path = os.path.join(path, "molecule_{}.png".format(i))
            file_path_true = os.path.join(path_true, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0], molecules[i][1])
            mol_true = self.mol_from_graphs(molecules_true[i][0], molecules_true[i][1])
            if mol is None or mol_true is None:
                logger.warning(f"Skipping invalid molecule pair at index {i}.")
                continue
            try:
                Draw.MolToFile(mol, file_path)
                Draw.MolToFile(mol_true, file_path_true)
            except rdkit.Chem.KekulizeException:
                logger.info("Can't kekulize molecule")

    def visualize_chain(self, batch_id, i, nodes_list, adjacency_matrix):
        path = os.path.join(self.result_path, f"chain/molecule_{batch_id}_{i}")
        os.makedirs(path, exist_ok=True)
        RDLogger.DisableLog("rdApp.*")
        mols = [
            self.mol_from_graphs(nodes_list[frame], adjacency_matrix[frame])
            for frame in range(len(nodes_list))
        ]
        if imageio is None:
            raise ImportError("imageio is required for diffusion-chain visualization.")
        valid_mols = [mol for mol in mols if mol is not None]
        if not valid_mols:
            logger.warning("Skipping diffusion chain containing no valid molecules.")
            return mols
        final_molecule = valid_mols[-1]
        AllChem.Compute2DCoords(final_molecule)
        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))
        for i, mol in enumerate(mols):
            if mol is None or mol.GetNumAtoms() != final_molecule.GetNumAtoms():
                continue
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))
        save_paths = []
        num_frames = len(nodes_list)
        for frame in range(num_frames):
            if mols[frame] is None:
                continue
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            Draw.MolToFile(
                mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}"
            )
            save_paths.append(file_name)
        if not save_paths:
            return mols
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(
            os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
        )
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        try:
            img = Draw.MolsToGridImage(
                [mol for mol in mols if mol is not None],
                molsPerRow=10,
                subImgSize=(200, 200),
            )
            img.save(
                os.path.join(path, "{}_grid_image.png".format(path.split("/")[-1]))
            )
        except Exception:
            logger.info("Can't kekulize molecule")
        return mols


def draw_volume(
    grid,
    density,
    atom_type,
    atom_coord,
    isomin=0.05,
    isomax=None,
    surface_count=5,
    title=None,
):
    if go is None:
        raise ImportError("plotly is required for volume visualization.")
    atom_colorscale = ["grey", "white", "red", "blue", "green"]

    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=grid[..., 0],
            y=grid[..., 1],
            z=grid[..., 2],
            value=density,
            isomin=isomin,
            isomax=isomax,
            opacity=0.1,
            surface_count=surface_count,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    axis_dict = dict(
        showgrid=False,
        showbackground=False,
        zeroline=False,
        visible=False,
    )

    fig.add_trace(
        go.Scatter3d(
            x=atom_coord[:, 0],
            y=atom_coord[:, 1],
            z=atom_coord[:, 2],
            mode="markers",
            marker=dict(
                size=10,
                color=atom_type,
                cmin=0,
                cmax=4,
                colorscale=atom_colorscale,
                opacity=0.6,
            ),
        )
    )

    if title is not None:
        title = dict(
            text=title,
            x=0.5,
            y=0.3,
            xanchor="center",
            yanchor="bottom",
        )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        showlegend=False,
        scene=dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict),
        title=title,
        title_font_family="Times New Roman",
    )

    return fig


def safe_write_image(fig, path, show_plot=False):
    path = Path(path)
    saved_path = None
    try:
        fig.write_image(path)
        saved_path = path
        logger.info(f"Image saved to: {path}")
    except Exception as e:
        logger.warning(f"Failed to save image {path}: {e}")
        try:
            html_path = path.with_suffix(".html")
            fig.write_html(html_path)
            saved_path = html_path
            logger.info(f"Saved interactive HTML instead: {html_path}")
        except Exception as html_e:
            logger.warning(f"Failed to save HTML fallback for {path}: {html_e}")

    if show_plot:
        try:
            if Image is None or display is None:
                raise ImportError("IPython is required to display image.")
            img_bytes = fig.to_image(format="png", scale=2)
            display(Image(img_bytes))
        except Exception as e:
            logger.warning(f"Failed to display image: {e}")
    return saved_path


def maybe_downsample_volume(grid, values, shape, max_points=250_000):
    """Downsample a regular volume for responsive Plotly rendering."""
    if shape is None:
        return grid, values, False, 1

    shape = tuple(int(size) for size in shape)
    if len(shape) != 3:
        return grid, values, False, 1
    total = math.prod(shape)
    if total != grid.shape[0] or any(val.shape[0] != grid.shape[0] for val in values):
        return grid, values, False, 1
    if total <= max_points:
        return grid, values, False, 1

    stride = max(1, math.ceil((total / max_points) ** (1 / 3)))
    grid_ds = grid.reshape(*shape, 3)[::stride, ::stride, ::stride].reshape(-1, 3)
    values_ds = [
        value.reshape(shape)[::stride, ::stride, ::stride].reshape(-1)
        for value in values
    ]
    return grid_ds, values_ds, True, stride
