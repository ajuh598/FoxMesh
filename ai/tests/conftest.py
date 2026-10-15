import numpy as np
import pytest

from foxmesh_ai.data.preprocess import PreprocessConfig, preprocess_dataset
from foxmesh_ai.data.synthetic import build_synthetic_dataset

FAST_PRE = PreprocessConfig(grid=48, uniform=2048, near=2048, surface=1024, render_size=32,
                            views=((0.0, 15.0), (40.0, 20.0)))


def sphere_field(res=32, r=0.6, center=(0, 0, 0)):
    c = (np.arange(res) + 0.5) / res * 2 - 1
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    return r - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    ds, _ = build_synthetic_dataset(root, 24, seed=3, res=40,
                                    families=["sword", "table", "mug", "tree", "snowman", "donut"])
    ds.resplit((0.75, 0.25, 0.0))
    preprocess_dataset(ds, FAST_PRE, workers=1)
    return ds
