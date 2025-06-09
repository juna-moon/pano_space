from tools.extract_occlusion import bbox_iou

def test_iou_full_overlap():
    a = [0,0,2,2]
    assert bbox_iou(a,a) == 1.0

def test_iou_partial():
    a = [0,0,2,2]
    b = [1,1,3,3]
    # 교차면적 = 1, union = 7 → IoU ≈0.142857
    assert round(bbox_iou(a,b),3) == 0.143