import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ─── 1) 상수 정의 ─────────────────────────────────────────────
IMAGE_HEIGHT = 2048
IMAGE_WIDTH  = 4096
WINDOW_SIZE  = 7
PATCH_SIZE   = 4
MULTIPLE     = WINDOW_SIZE * PATCH_SIZE  # 28

# 패딩 후 크기 계산 (고정)
pad_h = (MULTIPLE - IMAGE_HEIGHT % MULTIPLE) % MULTIPLE  # 24
pad_w = (MULTIPLE - IMAGE_WIDTH  % MULTIPLE) % MULTIPLE  # 16
PADDED_HEIGHT = IMAGE_HEIGHT + pad_h                   # 2072
PADDED_WIDTH  = IMAGE_WIDTH  + pad_w                   # 4112

# ─── 2) Dataset ────────────────────────────────────────────────
class PanoSet(Dataset):
    def __init__(self, list_file):
        with open(list_file) as f:
            self.paths = [l.strip() for l in f]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        # 4K 리사이징
        img = Image.open(self.paths[idx]).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        t   = torchvision.transforms.ToTensor()(img)
        # dummy mask
        mask = torch.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.long)
        return t, mask

# ─── 3) 패딩 헬퍼 (입력→PADDED 크기) ─────────────────────────
def pad_to_multiple(x, multiple=MULTIPLE):
    _,_,H,W = x.shape
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, H, W

# ─── 4) Decoder 블록 & U-Net 모델 ─────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UnetSwinSmall(nn.Module):
    def __init__(self, backbone_name='swin_small_patch4_window7_224', num_classes=2):
        super().__init__()
        # ── 5) 여기서 img_size를 (2072,4112)로 고정 ───────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(0,1,2,3),
            img_size=(PADDED_HEIGHT, PADDED_WIDTH),
        )
        chs = self.backbone.feature_info.channels()  # e.g. [96,192,384,768]

        self.up4 = DecoderBlock(chs[3] + chs[2], chs[2])
        self.up3 = DecoderBlock(chs[2] + chs[1], chs[1])
        self.up2 = DecoderBlock(chs[1] + chs[0], chs[0])
        self.up1 = DecoderBlock(chs[0],           chs[0])
        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        # ── 6) 빈 캐시 정리 (옵션)
        torch.cuda.empty_cache()
        # ── 7) 입력 패딩 → (2072,4112)
        x, H, W = pad_to_multiple(x)

        feats = self.backbone(x)
        x1, x2, x3, x4 = feats

        # ── 8) U-Net 디코더 (정확한 크기로 업샘플링)
        d4 = self.up4(torch.cat([
            F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False), x3
        ], dim=1))
        d3 = self.up3(torch.cat([
            F.interpolate(d4, size=x2.shape[2:], mode='bilinear', align_corners=False), x2
        ], dim=1))
        d2 = self.up2(torch.cat([
            F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=False), x1
        ], dim=1))
        d1 = self.up1(F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False))

        out = self.head(d1)
        # ── 9) 원본 크기로 자르기 → (2048,4096)
        return out[..., :H, :W]

# ─── 5) Training loop ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    args = ap.parse_args()

    net  = UnetSwinSmall(num_classes=2).cuda()
    opt  = torch.optim.Adam(net.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(PanoSet(args.train), batch_size=2, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(PanoSet(args.val),   batch_size=1, shuffle=False, num_workers=2)

    for epoch in range(5):
        net.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            pred = net(x)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader.dataset):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(net.state_dict(), "artifacts/baseline_unet.pth")
    print("Checkpoint saved → artifacts/baseline_unet.pth")

if __name__ == "__main__":
    main()