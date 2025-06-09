import os, glob, pytest

DATASETS = [
    "data/OSR_Bench/pano_4k",
    "data/stanford2d3ds/pano_4k",
    "data/OcclRobMV/pano_4k",
]

@pytest.mark.parametrize("dset", DATASETS)
def test_dataset_paths_exist(dset):
    assert os.path.isdir(dset)
    assert len(glob.glob(os.path.join(dset, "*.png"))) > 0