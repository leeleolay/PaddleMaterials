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

import io

import paddle
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image as PILImage

plt.switch_backend("agg")
cmap = ListedColormap(["grey", "white", "red", "blue", "green", "white"])


def draw_stack(density, atom_type=None, atom_coord=None, dim=-1):
    """Draw a 2-D projection of a density tensor and return an image array."""

    plt.figure(figsize=(3, 3))
    plt.imshow(density.sum(axis=dim).detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    if atom_type is not None:
        index = [axis for axis in range(3) if axis != dim % 3]
        coord = atom_coord.detach().cpu().numpy()
        color = cmap(atom_type.detach().cpu().numpy())
        plt.scatter(coord[:, index[1]], coord[:, index[0]], c=color, alpha=0.8)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpg")
    buffer.seek(0)
    image = PILImage.open(buffer)
    image = paddle.vision.transforms.ToTensor()(image)
    image = image.transpose([1, 2, 0])
    plt.close()
    return image.numpy()
