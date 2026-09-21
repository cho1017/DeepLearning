import os, sys, random, time, zipfile, shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', message='Conversion from CIE-LAB')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim
from thop import profile as thop_profile
from torchvision.ops import Conv2dNormActivation
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스 : {device}")
print(f"PyTorch 버전  : {torch.__version__}")

# ── 데이터셋 경로 설정 ─────────────────────────────────────────
DATA_ROOT = Path('./data/flicker8k/Images/archive/Images')
IMG_SIZE   = 128

# 경로 확인
if DATA_ROOT.exists():
    all_images = sorted([p for p in DATA_ROOT.glob('*.jpg')])
    print(f"✅ 이미지 발견: {len(all_images)}장  |  경로: {DATA_ROOT}")
else:
    print(f"❌ 데이터 경로를 찾을 수 없습니다: {DATA_ROOT}")
    print("   Kaggle에서 Flickr8k를 다운로드하고 아래 경로에 Images 폴더를 위치시켜 주세요:")
    print(f"   {DATA_ROOT.parent}")
    all_images = []

# ── LAB 색공간 시각화 ──────────────────────────────────────────
def show_lab_decomposition(img_path, ax_row):
    """RGB → LAB 분해 시각화"""
    img_rgb  = np.array(Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_lab  = rgb2lab(img_rgb)               # L: [0,100], ab: [-128,127]

    L  = img_lab[:,:,0]                       # 밝기
    a  = img_lab[:,:,1]                       # 초록↔빨강
    b  = img_lab[:,:,2]                       # 파랑↔노랑

    # ab → RGB 시각화용: L을 50으로 고정
    ab_vis        = np.full_like(img_lab, 50.0)
    ab_vis[:,:,1] = a
    ab_vis[:,:,2] = b

    ax_row[0].imshow(img_rgb);                           ax_row[0].set_title("원본 RGB")
    ax_row[1].imshow(L,  cmap='gray', vmin=0, vmax=100); ax_row[1].set_title("L 채널 (밝기, 모델 입력)")
    ax_row[2].imshow(a,  cmap='RdYlGn_r');               ax_row[2].set_title("a 채널 (초록↔빨강)")
    ax_row[3].imshow(b,  cmap='coolwarm');                ax_row[3].set_title("b 채널 (파랑↔노랑)")
    ax_row[4].imshow(lab2rgb(ab_vis.clip(-128,127)));    ax_row[4].set_title("ab 합성 (L=50 고정)")
    for ax in ax_row: ax.axis('off')

if len(all_images) >= 3:
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    for row, img_path in enumerate(all_images[:3]):
        show_lab_decomposition(img_path, axes[row])
    plt.suptitle("LAB 색공간 분해 — L(입력) / ab(예측 대상)", fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
else:
    print("이미지 로드 후 실행하세요.")

# ── 데이터셋 통계 분석 ─────────────────────────────────────────
if len(all_images) > 0:
    sample_n = min(300, len(all_images))
    a_vals, b_vals = [], []

    for p in random.sample(all_images, sample_n):
        img = np.array(Image.open(p).convert('RGB').resize((64,64))) / 255.0
        lab = rgb2lab(img)
        a_vals.extend(lab[:,:,1].flatten())
        b_vals.extend(lab[:,:,2].flatten())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(a_vals, bins=60, color='#e74c3c', alpha=0.7, edgecolor='white')
    axes[0].set_title("a 채널 분포 (초록↔빨강)"); axes[0].set_xlabel("픽셀 값"); axes[0].set_ylabel("빈도")
    axes[1].hist(b_vals, bins=60, color='#3498db', alpha=0.7, edgecolor='white')
    axes[1].set_title("b 채널 분포 (파랑↔노랑)"); axes[1].set_xlabel("픽셀 값"); axes[1].set_ylabel("빈도")
    plt.suptitle(f"Flickr8k ab 채널 픽셀 분포 ({sample_n}장 샘플)", fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()

    print(f"a 채널 — 평균: {np.mean(a_vals):+.2f}  std: {np.std(a_vals):.2f}  범위: [{min(a_vals):.1f}, {max(a_vals):.1f}]")
    print(f"b 채널 — 평균: {np.mean(b_vals):+.2f}  std: {np.std(b_vals):.2f}  범위: [{min(b_vals):.1f}, {max(b_vals):.1f}]")

class ColorizeAugment:
    """
    Colorization용 동기화 증강 클래스

    __call__(pil_image) → (L_tensor, ab_tensor)
      L_tensor  : (1, H, W)  float32,  범위 [0, 1]  (정규화: L/100)
      ab_tensor : (2, H, W)  float32,  범위 [-1, 1] (정규화: ab/110)
    """
    def __init__(self, img_size=128, augment=False):
        self.img_size = img_size
        self.augment  = augment

    def __call__(self, pil_img):
        # TODO 1: 리사이즈
        img_resized = pil_img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        # TODO 2: [augment=True] 랜덤 수평 반전 (50% 확률)
        if self.augment and random.random() > 0.5:
            img_resized = img_resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # TODO 3: [augment=True] 랜덤 크롭
        if self.augment and random.random() > 0.5:
            pad = self.img_size // 8
            padded_size = self.img_size + 2*pad
            img_resized = img_resized.resize((padded_size, padded_size), Image.BILINEAR)
            x0 = random.randint(0, 2*pad)
            y0 = random.randint(0, 2*pad)
            img_resized = img_resized.crop((x0, y0, x0+self.img_size, y0+self.img_size))

        # ── 이하 변경 금지 ─────────────────────────────────────
        # PIL → numpy → LAB 변환
        img_np  = np.array(img_resized.convert('RGB'), dtype=np.float32) / 255.0
        img_lab = rgb2lab(img_np).astype(np.float32)   # L:[0,100], ab:[-128,127]

        L  = img_lab[:, :, 0]    # (H, W)
        ab = img_lab[:, :, 1:]   # (H, W, 2)

        # TODO 4: [augment=True] L 채널 밝기 지터 (30% 확률)
        if self.augment and random.random() > 0.7:
            jitter = random.uniform(-10, 10)
            L = np.clip(L + jitter, 0, 100)

        # 정규화 후 Tensor 변환
        L_tensor  = torch.from_numpy(L / 100.0).unsqueeze(0)           # (1,H,W) [0,1]
        ab_tensor = torch.from_numpy(ab.transpose(2,0,1) / 110.0)      # (2,H,W) [-1,1]

        return L_tensor, ab_tensor


# ── 증강 파이프라인 검증 ──────────────────────────────────────
if len(all_images) > 0:
    aug_fn   = ColorizeAugment(img_size=IMG_SIZE, augment=True)
    noaug_fn = ColorizeAugment(img_size=IMG_SIZE, augment=False)

    sample_pil = Image.open(all_images[0])
    L_t, ab_t  = aug_fn(sample_pil)

    print("[shape 검증]")
    print(f"  L  tensor shape : {L_t.shape}")    # 기대: (1, 128, 128)
    print(f"  ab tensor shape : {ab_t.shape}")   # 기대: (2, 128, 128)
    print(f"  L  dtype        : {L_t.dtype}")    # 기대: float32
    print(f"  L  범위         : [{L_t.min():.3f}, {L_t.max():.3f}]")   # 기대: [0, 1]
    print(f"  ab 범위         : [{ab_t.min():.3f}, {ab_t.max():.3f}]") # 기대: [-1, 1] 내외

    assert L_t.shape  == (1, IMG_SIZE, IMG_SIZE), "❌ L shape 오류"
    assert ab_t.shape == (2, IMG_SIZE, IMG_SIZE), "❌ ab shape 오류"
    assert L_t.dtype  == torch.float32,           "❌ dtype은 float32 이어야 합니다"
    assert 0.0 <= L_t.min() and L_t.max() <= 1.0, "❌ L 범위는 [0,1] 이어야 합니다"
    print("\n✅ ColorizeAugment 검증 통과")

class Flickr8kColorizationDataset(Dataset):
    """
    Flickr8k Colorization Dataset

    Args:
        image_paths  : list of Path  — 이미지 파일 경로 리스트
        augment_fn   : ColorizeAugment 인스턴스
    Returns:
        (L_tensor, ab_tensor)
          L_tensor  : (1, H, W)  [0, 1]
          ab_tensor : (2, H, W)  [-1, 1] 내외
    """
    def __init__(self, image_paths, augment_fn):
        self.image_paths = image_paths
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx])
            img_rgb = img.convert('RGB')
            L_tensor, ab_tensor = self.augment_fn(img_rgb)
            return L_tensor, ab_tensor
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


# ── 데이터셋 분할 & DataLoader 생성 ──────────────────────────
if len(all_images) > 0:
    random.shuffle(all_images)
    n_total = len(all_images)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_paths = all_images[:n_train]
    val_paths   = all_images[n_train:n_train + n_val]
    test_paths  = all_images[n_train + n_val:]

    aug_train = ColorizeAugment(img_size=IMG_SIZE, augment=True)
    aug_eval  = ColorizeAugment(img_size=IMG_SIZE, augment=False)

    train_dataset = Flickr8kColorizationDataset(train_paths, augment_fn=aug_train)
    val_dataset   = Flickr8kColorizationDataset(val_paths,   augment_fn=aug_eval)
    test_dataset  = Flickr8kColorizationDataset(test_paths,  augment_fn=aug_eval)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train : {len(train_dataset):,}장  ({len(train_loader)} 배치)")
    print(f"Val   : {len(val_dataset):,}장  ({len(val_loader)} 배치)")
    print(f"Test  : {len(test_dataset):,}장  ({len(test_loader)} 배치)")

# ── Dataset 검증 ─────────────────────────────────────────────
if len(all_images) > 0:
    L_t, ab_t = train_dataset[0]

    print("[shape 검증]")
    print(f"  L  shape : {L_t.shape}")    # 기대: (1, 128, 128)
    print(f"  ab shape : {ab_t.shape}")   # 기대: (2, 128, 128)
    print(f"  L  dtype : {L_t.dtype}")    # 기대: float32
    print(f"  ab dtype : {ab_t.dtype}")   # 기대: float32

    assert L_t.shape  == (1, IMG_SIZE, IMG_SIZE), "❌ L shape 오류"
    assert ab_t.shape == (2, IMG_SIZE, IMG_SIZE), "❌ ab shape 오류"
    assert L_t.dtype  == torch.float32,           "❌ dtype 오류"
    print("\n✅ Dataset shape 검증 통과")

    # 배치 shape 확인
    L_batch, ab_batch = next(iter(train_loader))
    print(f"\n[배치 shape]")
    print(f"  L  batch : {L_batch.shape}")   # 기대: (32, 1, 128, 128)
    print(f"  ab batch : {ab_batch.shape}")  # 기대: (32, 2, 128, 128)

def lab_tensor_to_rgb(L_tensor, ab_tensor):
    """
    (1,H,W) L + (2,H,W) ab 텐서 → RGB numpy 이미지 (H,W,3) [0,1]
    L_tensor  : [0,1]    → 역정규화 × 100
    ab_tensor : [-1,1]   → 역정규화 × 110
    """
    L_np  = L_tensor.squeeze().cpu().numpy() * 100.0
    ab_np = ab_tensor.cpu().numpy().transpose(1,2,0) * 110.0
    lab   = np.concatenate([L_np[:,:,None], ab_np], axis=2)
    return lab2rgb(lab.clip(-128, 127)).clip(0, 1)


def batch_psnr_colorization(pred_ab, target_ab, L_batch):
    """
    배치 평균 PSNR 계산
    pred_ab, target_ab : (B,2,H,W)
    L_batch            : (B,1,H,W)
    """
    psnr_list = []
    B = pred_ab.shape[0]
    for i in range(B):
        pred_rgb   = lab_tensor_to_rgb(L_batch[i], pred_ab[i])
        target_rgb = lab_tensor_to_rgb(L_batch[i], target_ab[i])
        psnr_list.append(calc_psnr(target_rgb, pred_rgb, data_range=1.0))
    return float(np.mean(psnr_list))


def batch_ssim_colorization(pred_ab, target_ab, L_batch, n_samples=4):
    """배치에서 n_samples장 샘플 SSIM 평균 계산"""
    ssim_list = []
    for i in range(min(n_samples, pred_ab.shape[0])):
        pred_rgb   = lab_tensor_to_rgb(L_batch[i], pred_ab[i])
        target_rgb = lab_tensor_to_rgb(L_batch[i], target_ab[i])
        ssim_list.append(calc_ssim(target_rgb, pred_rgb, data_range=1.0, channel_axis=2))
    return float(np.mean(ssim_list))

print("유틸리티 함수 정의 완료")


class BaselineColorizer(nn.Module):
    """
    베이스라인 Encoder-Decoder Colorization 모델
    입력 : (B, 1, 128, 128)  L 채널
    출력 : (B, 2, 128, 128)  ab 채널 예측, 범위 [-1, 1]
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, padding=0, bias=False, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, padding=0, bias=False, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, padding=0, bias=False, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.bottleneck(self.encoder(x)))


# ── shape 검증 ─────────────────────────────────────────────────
baseline_model = BaselineColorizer().to(device)
dummy = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE).to(device)
out   = baseline_model(dummy)

print(f"입력  shape : {dummy.shape}")
print(f"출력  shape : {out.shape}")       # 기대: (2, 2, 128, 128)
assert out.shape == (2, 2, IMG_SIZE, IMG_SIZE), "❌ 출력 shape 오류"
assert out.min() >= -1 and out.max() <= 1,       "❌ 출력은 [-1,1] 범위여야 합니다 (Tanh 확인)"
print("✅ BaselineColorizer 검증 통과")

base_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
print(f"\nBaseline 파라미터 수 : {base_params:,}")


# ─────────────────────────────────────────────────────────────
# Attention Gate
#
# skip connection(e3, e2)을 decoder로 그대로 넘기기 전에,
# "지금 decoder가 복원 중인 문맥(g)"과 비교해서
# encoder feature(x)에서 관련 있는 위치만 강조(0~1 가중치)하는 모듈.
#   g : decoder 쪽 gating 신호 (더 깊고 coarse한 문맥)
#   x : encoder 쪽 skip 신호   (디테일은 많지만 노이즈도 섞임)
# 반드시 g와 x의 (H, W)는 같아야 함 (채널 수는 달라도 됨).
# ─────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.relu(g1 + x1)
        alpha = self.psi(alpha)      # (B, 1, H, W), 0~1 사이 중요도 지도
        return x * alpha             # 채널마다 같은 alpha가 곱해짐 (broadcasting)


class ImprovedColorizer(nn.Module):
    """
    개선된 Colorization 모델 — Residual bottleneck + Skip connection + Attention Gate

    조건:
      - 입력 : (B, 1, 128, 128)
      - 출력 : (B, 2, 128, 128), 범위 [-1, 1] (Tanh)
      - 베이스라인보다 높은 PSNR 달성
    """
    def __init__(self):
        super().__init__()

        def block1(inp, out, stride):
            return nn.Sequential(
                nn.Conv2d(inp, out, 3, padding=1, bias=False, stride=stride),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
            )
        def block2(inp, out, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(inp, out, 2, padding=0, bias=False, stride=stride),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block1(1, 64, 1)
        self.enc2 = block1(64, 128, 2)
        self.enc3 = block1(128, 256, 2)
        self.enc4 = block1(256, 512, 2)
        self.bottleneck = nn.Sequential(
            block1(512, 512, 1),
            block1(512, 512, 1),
            block1(512, 512, 1),

        )
        self.dec1 = block2(1024, 256, 2)
        self.dec2 = block2(512, 128, 2)
        self.dec3 = block2(256, 64, 2)
        self.dec4 = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # ── Attention Gate 추가 ──────────────────────────────
        # d1(256,32,32)이 gating 신호, e3(256,32,32)가 skip 신호
        self.ag2 = AttentionGate(F_g=256, F_x=256, F_int=128)
        # d2(128,64,64)가 gating 신호, e2(128,64,64)가 skip 신호
        self.ag3 = AttentionGate(F_g=128, F_x=128, F_int=64)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bt = self.bottleneck(e4)        # 16x16
        bt = bt + e4                    # residual
        up1 = torch.cat([e4, bt], dim=1)
        d1 = self.dec1(up1)             # 32x32

        e3_att = self.ag2(g=d1, x=e3)   # e3를 attention으로 거름
        up2 = torch.cat([d1, e3_att], dim=1)
        d2 = self.dec2(up2)             # 64x64

        e2_att = self.ag3(g=d2, x=e2)   # e2를 attention으로 거름
        up3 = torch.cat([d2, e2_att], dim=1)
        d3 = self.dec3(up3)             # 128x128

        x1 = self.dec4(d3)
        x = self.tanh(x1)
        return x


# ── shape 검증 ─────────────────────────────────────────────────
improved_model = ImprovedColorizer().to(device)
dummy = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE).to(device)
out   = improved_model(dummy)

print(f"입력  shape : {dummy.shape}")
print(f"출력  shape : {out.shape}")        # 기대: (2, 2, 128, 128)
assert out.shape == (2, 2, IMG_SIZE, IMG_SIZE), "❌ 출력 shape 오류"
assert out.min() >= -1 and out.max() <= 1,       "❌ 출력은 [-1,1] 범위 (Tanh 확인)"
print("✅ ImprovedColorizer 검증 통과")

imp_params = sum(p.numel() for p in improved_model.parameters() if p.requires_grad)
print(f"\nImproved  파라미터 수 : {imp_params:,}")
print(f"Baseline  파라미터 수 : {base_params:,}")


def train_one_epoch(model, loader, criterion, optimizer, epoch, writer, tag):
    """한 epoch 학습 후 TensorBoard에 Loss / PSNR 기록"""
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    for L_batch, ab_batch in loader:
        L_batch  = L_batch.to(device)    # (B,1,H,W)
        ab_batch = ab_batch.to(device)   # (B,2,H,W)  ← 타깃

        # TODO 14: 학습 한 스텝 구현
        # 순서: zero_grad → forward(L_batch) → loss(pred_ab, ab_batch) → backward → step
        # 변수명: pred_ab = model(L_batch)
        optimizer.zero_grad()
        pred_ab = model(L_batch)
        loss = criterion(pred_ab, ab_batch)
        loss.backward()
        optimizer.step()
        pass  # TODO: 이 줄을 지우고 구현하세요

        with torch.no_grad():
            total_psnr += batch_psnr_colorization(
                pred_ab.detach().cpu(), ab_batch.cpu(), L_batch.cpu())
        total_loss += loss.item()
        n += 1

    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    writer.add_scalar(f'{tag}/Loss/train', avg_loss, epoch)
    writer.add_scalar(f'{tag}/PSNR/train', avg_psnr, epoch)
    return avg_loss, avg_psnr


def evaluate(model, loader, criterion, epoch, writer, tag):
    """검증/테스트 평가 후 TensorBoard에 기록"""
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    all_ssim = []

    with torch.no_grad():
        for L_batch, ab_batch in loader:
            L_batch  = L_batch.to(device)
            ab_batch = ab_batch.to(device)

            # TODO 15: 평가 한 스텝 구현

            pred_ab = model(L_batch)
            loss = criterion(pred_ab, ab_batch)
            # (backward, optimizer.step 불필요)
            pass  # TODO: 이 줄을 지우고 구현하세요

            total_loss += loss.item()
            total_psnr += batch_psnr_colorization(
                pred_ab.cpu(), ab_batch.cpu(), L_batch.cpu())
            all_ssim.append(batch_ssim_colorization(
                pred_ab.cpu(), ab_batch.cpu(), L_batch.cpu()))
            n += 1

    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    avg_ssim = float(np.mean(all_ssim))
    writer.add_scalar(f'{tag}/Loss/val',  avg_loss, epoch)
    writer.add_scalar(f'{tag}/PSNR/val',  avg_psnr, epoch)
    writer.add_scalar(f'{tag}/SSIM/val',  avg_ssim, epoch)
    return avg_loss, avg_psnr, avg_ssim

def run_training(model, tag, epochs=30, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer    = SummaryWriter(log_dir=f'runs/{tag}')

    writer.add_graph(model, torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(device))

    history = {'train_loss':[], 'val_loss':[], 'train_psnr':[], 'val_psnr':[], 'val_ssim':[]}

    print(f"\n{'='*68}")
    print(f"  {tag} 학습 시작 | {epochs} epochs | device: {device}")
    print(f"{'='*68}")
    print(f"{'Epoch':>6} | {'TrLoss':>7} | {'TrPSNR':>7} | {'VaLoss':>7} | {'VaPSNR':>7} | {'VaSSIM':>7}")
    print(f"{'-'*68}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_psnr          = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, tag)
        va_loss, va_psnr, va_ssim = evaluate(model, val_loader, criterion, epoch, writer, tag)
        scheduler.step()

        history['train_loss'].append(tr_loss); history['train_psnr'].append(tr_psnr)
        history['val_loss'].append(va_loss);   history['val_psnr'].append(va_psnr)
        history['val_ssim'].append(va_ssim)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {tr_loss:>7.4f} | {tr_psnr:>7.2f} | {va_loss:>7.4f} | {va_psnr:>7.2f} | {va_ssim:>7.4f}")

        # TensorBoard 이미지 기록 (5 epoch마다)
        if epoch % 5 == 0:
            _log_colorization_images(model, val_loader, writer, tag, epoch)

    elapsed = time.time() - start
    writer.close()
    print(f"\n완료 ({elapsed:.0f}초) | 최종 Val PSNR: {history['val_psnr'][-1]:.2f} dB  SSIM: {history['val_ssim'][-1]:.4f}")
    return history


def _log_colorization_images(model, loader, writer, tag, epoch, n=6):
    """TensorBoard IMAGES 탭에 채색 결과 기록"""
    model.eval()
    L_b, ab_b = next(iter(loader))
    L_dev = L_b[:n].to(device)
    with torch.no_grad():
        pred_ab = model(L_dev).cpu()

    imgs_gray, imgs_pred, imgs_gt = [], [], []
    for i in range(n):
        gray = L_b[i].repeat(3,1,1)
        pred = torch.tensor(lab_tensor_to_rgb(L_b[i], pred_ab[i])).permute(2,0,1).float()
        gt   = torch.tensor(lab_tensor_to_rgb(L_b[i], ab_b[i])).permute(2,0,1).float()
        imgs_gray.append(gray); imgs_pred.append(pred); imgs_gt.append(gt)

    writer.add_image(f'{tag}/gray',     vutils.make_grid(imgs_gray, nrow=n), epoch)
    writer.add_image(f'{tag}/pred_rgb', vutils.make_grid(imgs_pred, nrow=n), epoch)
    writer.add_image(f'{tag}/gt_rgb',   vutils.make_grid(imgs_gt,   nrow=n), epoch)

print("학습 함수 정의 완료")

# ── Baseline 학습 ────────────────────────────────────────────
base_history = run_training(baseline_model, tag='Baseline', epochs=30)

# ── ImprovedColorizer 학습 ───────────────────────────────────
imp_history = run_training(improved_model, tag='Improved', epochs=30)

# ── FLOPs & Params & Latency 측정 ───────────────────────────
def get_model_stats(model, input_size=(1, 1, 128, 128)):
    dummy = torch.zeros(input_size).to(device)
    model.eval()
    flops, params = thop_profile(model, inputs=(dummy,), verbose=False)
    with torch.no_grad():
        for _ in range(10): model(dummy)          # warm-up
        t0 = time.time()
        for _ in range(100): model(dummy)
        latency_ms = (time.time() - t0) / 100 * 1000
    return flops, params, latency_ms

base_flops,  base_params_cnt,  base_lat  = get_model_stats(baseline_model)
imp_flops,   imp_params_cnt,   imp_lat   = get_model_stats(improved_model)

print(f"{'지표':<22} | {'Baseline':>14} | {'Improved':>14}")
print("-" * 55)
print(f"{'Params':<22} | {base_params_cnt:>14,.0f} | {imp_params_cnt:>14,.0f}")
print(f"{'FLOPs':<22} | {base_flops:>14,.0f} | {imp_flops:>14,.0f}")
print(f"{'FLOPs (GMACs)':<22} | {base_flops/1e9:>13.3f}G | {imp_flops/1e9:>13.3f}G")
print(f"{'Latency (ms)':<22} | {base_lat:>13.3f}  | {imp_lat:>13.3f}")

# ── 테스트셋 최종 성능 측정 ──────────────────────────────────
dummy_writer = SummaryWriter('/tmp/dummy_eval')

base_te_loss, base_te_psnr, base_te_ssim = evaluate(
    baseline_model, test_loader, nn.MSELoss(), 0, dummy_writer, 'dummy')
imp_te_loss,  imp_te_psnr,  imp_te_ssim  = evaluate(
    improved_model, test_loader, nn.MSELoss(), 0, dummy_writer, 'dummy')
dummy_writer.close()

print("\n" + "="*70)
print(f"  {'모델':<22} | {'PSNR':>8} | {'SSIM':>7} | {'Params':>12} | {'FLOPs':>10}")
print("-"*70)
print(f"  {'Baseline':<22} | {base_te_psnr:>7.2f}dB | {base_te_ssim:>7.4f} | {base_params_cnt:>12,.0f} | {base_flops/1e6:>8.1f}M")
print(f"  {'Improved':<22} | {imp_te_psnr:>7.2f}dB | {imp_te_ssim:>7.4f} | {imp_params_cnt:>12,.0f} | {imp_flops/1e6:>8.1f}M")
print("="*70)
psnr_gain = imp_te_psnr - base_te_psnr
print(f"\n  PSNR 향상 : {psnr_gain:+.2f} dB  ({'✅ 개선 달성' if psnr_gain > 0 else '❌ 미달성 — 모델을 재설계하세요'})")

# ── 종합 비교 차트 (6종) ─────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
epochs_r = range(1, 31)

# 1) Loss 곡선
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(epochs_r, base_history['val_loss'], 'o-', label='Baseline', color='#e74c3c', lw=2)
ax1.plot(epochs_r, imp_history['val_loss'],  's-', label='Improved',  color='#2980b9', lw=2)
ax1.set_title('Val Loss (MSE) ↓', fontweight='bold'); ax1.set_xlabel('Epoch')
ax1.legend(); ax1.grid(True, alpha=0.3)

# 2) PSNR 곡선
ax2 = fig.add_subplot(gs[0,1])
ax2.plot(epochs_r, base_history['val_psnr'], 'o-', label='Baseline', color='#e74c3c', lw=2)
ax2.plot(epochs_r, imp_history['val_psnr'],  's-', label='Improved',  color='#2980b9', lw=2)
ax2.set_title('Val PSNR (dB) ↑', fontweight='bold'); ax2.set_xlabel('Epoch')
ax2.legend(); ax2.grid(True, alpha=0.3)

# 3) SSIM 곡선
ax3 = fig.add_subplot(gs[0,2])
ax3.plot(epochs_r, base_history['val_ssim'], 'o-', label='Baseline', color='#e74c3c', lw=2)
ax3.plot(epochs_r, imp_history['val_ssim'],  's-', label='Improved',  color='#2980b9', lw=2)
ax3.set_title('Val SSIM ↑', fontweight='bold'); ax3.set_xlabel('Epoch')
ax3.legend(); ax3.grid(True, alpha=0.3)

# 4) Params 비교
ax4 = fig.add_subplot(gs[1,0])
vals = [base_params_cnt/1e6, imp_params_cnt/1e6]
bars = ax4.bar(['Baseline','Improved'], vals, color=['#e74c3c','#2980b9'], width=0.5)
ax4.set_title('Parameters (M) ↓', fontweight='bold'); ax4.set_ylabel('Millions')
for b, v in zip(bars, vals):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.01,
             f'{v:.2f}M', ha='center', fontsize=10, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5) FLOPs 비교
ax5 = fig.add_subplot(gs[1,1])
fvals = [base_flops/1e9, imp_flops/1e9]
bars2 = ax5.bar(['Baseline','Improved'], fvals, color=['#e74c3c','#2980b9'], width=0.5)
ax5.set_title('FLOPs (GMACs) ↓', fontweight='bold'); ax5.set_ylabel('GMACs')
for b, v in zip(bars2, fvals):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+max(fvals)*0.01,
             f'{v:.2f}G', ha='center', fontsize=10, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6) PSNR vs Params 트레이드오프
ax6 = fig.add_subplot(gs[1,2])
ax6.scatter([base_params_cnt/1e6], [base_te_psnr], s=250, color='#e74c3c',
            zorder=5, label=f'Baseline\n({base_te_psnr:.2f}dB)')
ax6.scatter([imp_params_cnt/1e6],  [imp_te_psnr],  s=250, color='#2980b9',
            zorder=5, label=f'Improved\n({imp_te_psnr:.2f}dB)')
ax6.set_xlabel('Parameters (M)'); ax6.set_ylabel('PSNR (dB)')
ax6.set_title('PSNR vs Params 트레이드오프', fontweight='bold')
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)

plt.suptitle('Baseline vs ImprovedColorizer — 종합 비교 (Flickr8k)', fontsize=14, fontweight='bold')
plt.show()

# ── 채색 결과 정성 비교 ──────────────────────────────────────
baseline_model.eval(); improved_model.eval()
L_batch, ab_batch = next(iter(test_loader))
L_dev = L_batch[:6].to(device)

with torch.no_grad():
    base_pred  = baseline_model(L_dev).cpu()
    imp_pred   = improved_model(L_dev).cpu()

fig, axes = plt.subplots(4, 6, figsize=(20, 14))
row_labels = ["흑백 입력 (L)", "Baseline 채색", "Improved 채색", "정답 (원본)"]

for col in range(6):
    gray_rgb = L_batch[col].repeat(3,1,1).permute(1,2,0).numpy()
    base_rgb = lab_tensor_to_rgb(L_batch[col], base_pred[col])
    imp_rgb  = lab_tensor_to_rgb(L_batch[col], imp_pred[col])
    gt_rgb   = lab_tensor_to_rgb(L_batch[col], ab_batch[col])

    axes[0,col].imshow(gray_rgb, cmap='gray'); axes[0,col].axis('off')
    axes[1,col].imshow(base_rgb);               axes[1,col].axis('off')
    axes[2,col].imshow(imp_rgb);                axes[2,col].axis('off')
    axes[3,col].imshow(gt_rgb);                 axes[3,col].axis('off')

    base_p = calc_psnr(gt_rgb, base_rgb, data_range=1.0)
    imp_p  = calc_psnr(gt_rgb, imp_rgb,  data_range=1.0)
    axes[1,col].set_title(f"PSNR:{base_p:.1f}dB", fontsize=8)
    axes[2,col].set_title(f"PSNR:{imp_p:.1f}dB",  fontsize=8,
                          color='green' if imp_p > base_p else 'red')

for ax, label in zip(axes[:,0], row_labels):
    ax.set_ylabel(label, fontsize=10, fontweight='bold')

plt.suptitle("채색 결과 정성 비교 (테스트셋)", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

