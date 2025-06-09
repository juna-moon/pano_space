import os, torch
from data_loaders.stanford2d3ds import Stanford2DDataset
from data_loaders.muva_ais     import MUVAISDataset
from data_loaders.capture  import CaptureDataset

def _basic_checks(ds):
    assert len(ds) > 0
    x = ds[0]                       # 한 샘플
    assert isinstance(x["rgb"], torch.Tensor)
    assert x["rgb"].shape[0] == 3   # (C,H,W) 채널 수
    assert x["rgb"].dtype == torch.float32
    # 선택 필드
    if "view" in x:   assert x["view"] in ("pano", "pinhole")

# tests/test_dataloaders.py
def test_stanford():
    ds  = Stanford2DDataset(os.getenv("STANFORD_IDX"))
    _basic_checks(ds)

def test_muva():
    ds = MUVAISDataset(os.getenv("MUVA_IDX"), split="train")
    _basic_checks(ds)

def test_capture():
    ds = CaptureDataset(os.getenv("CAP_IDX"), subset="synthetic")
    _basic_checks(ds)