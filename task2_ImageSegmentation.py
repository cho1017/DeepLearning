import os
import time

import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboard.compat import tf
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 윈도우 기본 한글 폰트(맑은 고딕) 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
# 재현성 고정
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스 : {device}")
print(f"PyTorch 버전  : {torch.__version__}")

# 데이터셋 다운로드 (torchvision 제공)
# target_types='segmentation' 으로 trimap 마스크도 함께 다운로드
raw_dataset = torchvision.datasets.OxfordIIITPet(
    root='./data',
    split='trainval',
    target_types='segmentation',
    download=True
)
print(f"전체 데이터 수 : {len(raw_dataset)}")

# 원본 이미지 & 마스크 확인 (변환 없이)
sample_img, sample_mask = raw_dataset[0]
print(f"원본 이미지 타입   : {type(sample_img)}")
print(f"원본 마스크 타입   : {type(sample_mask)}")
print(f"원본 이미지 크기   : {sample_img.size}")

print(f"원본 마스크 크기   : {sample_mask.size}")
print(f"마스크 고유 픽셀 값: {np.unique(np.array(sample_mask))}")  # [1, 2, 3] 확인

# 데이터셋 샘플 시각화 (원본)
TRIMAP_COLORS = {
    1: ([0.2, 0.6, 1.0], 'Foreground'),
    2: ([0.9, 0.9, 0.9], 'Background'),
    3: ([1.0, 0.5, 0.0], 'Not classified'),
}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# 0=Foreground, 1=Background, 2=Not classified (0-indexed 기준)
LABEL_COLORS = np.array([
    [0.2, 0.6, 1.0],   # Foreground
    [0.9, 0.9, 0.9],   # Background
    [1.0, 0.5, 0.0],   # Not classified
])
def denorm(img_tensor):
    """정규화된 이미지 텐서를 화면에 보여줄 수 있는 (H,W,3) 형태로 되돌림"""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img = img_tensor * std + mean          # 정규화 역연산: (x*std)+mean
    img = img.clamp(0, 1)                  # 계산 오차로 0~1 살짝 벗어난 값 보정
    return img.permute(1, 2, 0).numpy()    # (C,H,W) → (H,W,C)로 변환 (matplotlib용)

def mask_to_rgb(mask_tensor):
    """클래스 인덱스(0/1/2) 마스크를 색깔 있는 RGB 이미지로 변환"""
    return LABEL_COLORS[mask_tensor.numpy()]
def visualize_sample(dataset, indices, title="Oxford-IIIT Pet 샘플"):
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        mask_arr = np.array(mask)
        mask_rgb = np.zeros((*mask_arr.shape, 3))
        for val, (color, _) in TRIMAP_COLORS.items():
            mask_rgb[mask_arr == val] = color

        axes[row, 0].imshow(img); axes[row, 0].set_title(f"원본 이미지 (idx={idx})")
        axes[row, 1].imshow(mask_rgb); axes[row, 1].set_title("Trimap 마스크")
        axes[row, 2].imshow(img); axes[row, 2].imshow(mask_rgb, alpha=0.5)
        axes[row, 2].set_title("오버레이")
        for ax in axes[row]: ax.axis('off')

    patches = [mpatches.Patch(color=c, label=l) for _, (c, l) in TRIMAP_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10)
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

visualize_sample(raw_dataset, indices=[0, 100, 500])

# 마스크 픽셀 분포 분석
print("마스크 픽셀 값 분포 분석 (처음 200장)\n")
pixel_counts = {1: 0, 2: 0, 3: 0}
for i in range(200):
    _, mask = raw_dataset[i]
    arr = np.array(mask).flatten()
    for v in [1, 2, 3]:
        pixel_counts[v] += (arr == v).sum()

total = sum(pixel_counts.values())
labels = ['Foreground (1)', 'Background (2)', 'Not classified (3)']
counts = [pixel_counts[1], pixel_counts[2], pixel_counts[3]]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(labels, counts, color=['#3498db','#bdc3c7','#e67e22'])
axes[0].set_title("픽셀 값 절대 수"); axes[0].set_ylabel("픽셀 수")
axes[1].pie([c/total for c in counts], labels=labels,
            colors=['#3498db','#bdc3c7','#e67e22'], autopct='%1.1f%%')
axes[1].set_title("픽셀 값 비율")
plt.suptitle("마스크 픽셀 분포 (200장 기준)", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

for l, c in zip(labels, counts):
    print(f"  {l}: {c:,}  ({100*c/total:.1f}%)")


class PetSegDataset(Dataset):
    """
    Oxford-IIIT Pet Segmentation Dataset

    Args:
        root_dataset : torchvision.datasets.OxfordIIITPet 인스턴스
        img_size     : 리사이즈 크기 (정사각형)
        augment      : 훈련용 데이터 증강 여부
    """

    # 마스크 픽셀 값 → 학습 레이블 매핑
    # 원본: 1=Foreground, 2=Background, 3=Not classified
    # 변환: 0=Foreground, 1=Background, 2=Not classified
    MASK_REMAP = {1: 0, 2: 1, 3: 2}
    NUM_CLASSES = 3
    CLASS_NAMES = ['Foreground', 'Background', 'Not classified']

    def __init__(self, root_dataset, img_size=128, augment=False):
        self.dataset  = root_dataset
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        # TODO 1: 데이터셋의 전체 길이를 반환하세요
        # 힌트: self.dataset의 길이를 반환
        return len(self.dataset)
        pass

    def __getitem__(self, idx):
        # TODO 2: idx번째 (이미지, 마스크) 쌍을 전처리하여 반환하세요
        #
        # 처리 순서:
        #   (1) self.dataset에서 PIL 이미지와 마스크를 가져온다
        #   (2) 이미지와 마스크를 img_size × img_size 로 리사이즈한다
        #       - 이미지: TF.resize(img, [self.img_size, self.img_size])
        #       - 마스크: TF.resize(mask, [self.img_size, self.img_size],
        #                           interpolation=TF.InterpolationMode.NEAREST)
        #         ※ 마스크는 NEAREST 보간 필수 (픽셀 값이 변형되면 안 됨)
        #
        #   (3) augment=True일 때 랜덤 좌우 반전 적용
        #       - 이미지와 마스크에 동일한 flip을 적용해야 함!
        #       - 힌트: r = random.random()으로 같은 난수를 공유
        #               if r > 0.5: img, mask = TF.hflip(img), TF.hflip(mask)
        #
        #   (4) 이미지를 Tensor로 변환하고 정규화한다
        #       - TF.to_tensor(img)  →  (3, H, W) float [0,1]
        #       - TF.normalize(img_tensor, mean=[0.485,0.456,0.406],
        #                                  std =[0.229,0.224,0.225])
        #
        #   (5) 마스크를 Long Tensor로 변환하고 픽셀 값을 0-indexed로 변환한다
        #       - np.array(mask) 로 numpy 변환
        #       - 픽셀 값 1/2/3 → 0/1/2 로 변환 (mask_arr - 1)
        #       - torch.from_numpy(mask_arr).long()
        #
        #   (6) (img_tensor, mask_tensor) 튜플을 반환한다
        img, mask = self.dataset[idx]
        img = TF.resize(img, [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size],interpolation=TF.InterpolationMode.NEAREST)
        r = random.random()

        if self.augment and r>0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=[0.485,0.456,0.406],
                                std =[0.229,0.224,0.225])
        mask_arr =np.array(mask)
        mask_arr -= 1
        mask_ten = torch.from_numpy(mask_arr).long()

        return  img_tensor, mask_ten


# ── 데이터셋 분할 ────────────────────────────────────────────
raw_all = torchvision.datasets.OxfordIIITPet(
    root='./data', split='trainval',
    target_types='segmentation', download=False
)

full_dataset = PetSegDataset(raw_all, img_size=128, augment=False)

# 7:1.5:1.5 비율로 train/val/test 분할
n_total = len(full_dataset)
n_train = int(n_total * 0.7)
n_val   = int(n_total * 0.15)
n_test  = n_total - n_train - n_val

train_base, val_dataset, test_dataset = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

# 훈련셋에만 augment 적용 (Subset은 transform 교체 불가 → 별도 Dataset 생성)
raw_train_subset = torch.utils.data.Subset(raw_all, train_base.indices)
train_dataset = PetSegDataset(raw_train_subset, img_size=128, augment=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train : {len(train_dataset):,}장  ({len(train_loader)} 배치)")
print(f"Val   : {len(val_dataset):,}장  ({len(val_loader)} 배치)")
print(f"Test  : {len(test_dataset):,}장  ({len(test_loader)} 배치)")

def double_conv(in_ch, out_ch, dropout_p=0.0):
    """Conv → BN → ReLU → Conv → BN → ReLU (+ 선택적 Dropout)"""
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout_p > 0:
        layers.append(nn.Dropout2d(p=dropout_p))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(UNet, self).__init__()

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64,  128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────
        self.bottleneck = double_conv(512, 1024)

        # ── Decoder ──────────────────────────────────────────
        # ConvTranspose2d : 해상도 2배 업샘플링
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512, dropout_p=0.2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256, dropout_p=0.2)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128, dropout_p=0.1)  # 출력에 가까울수록 약하게

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64, dropout_p=0.1)

        # ── 최종 출력 (1×1 Conv) ─────────────────────────────
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                   # (B,  64, 128, 128)
        e2 = self.enc2(self.pool(e1))        # (B, 128,  64,  64)
        e3 = self.enc3(self.pool(e2))        # (B, 256,  32,  32)
        e4 = self.enc4(self.pool(e3))        # (B, 512,  16,  16)

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))  # (B,1024,   8,   8)

        # Decoder (Up + Skip Connection + double_conv)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))   # (B, 512, 16, 16)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # (B, 256, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # (B, 128, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # (B,  64,128,128)

        return self.out_conv(d1)                                 # (B,   3,128,128)


# ── 모델 검증 ──────────────────────────────────────────────────
model = UNet(in_channels=3, num_classes=PetSegDataset.NUM_CLASSES).to(device)

dummy  = torch.zeros(2, 3, 128, 128).to(device)
output = model(dummy)
print(f"입력  shape : {dummy.shape}")
print(f"출력  shape : {output.shape}")    # 기대: (2, 3, 128, 128)
assert output.shape == (2, 3, 128, 128), "❌ U-Net 출력 shape 오류"
print("✅ U-Net shape 검증 통과")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n학습 파라미터 수 : {total_params:,}")

def compute_iou(preds, targets, num_classes=3):
    """
    배치 단위 IoU 계산
    preds   : (B, C, H, W) logits
    targets : (B, H, W) long
    반환    : 클래스별 IoU list, mIoU float
    """
    pred_labels = preds.argmax(dim=1)   # (B, H, W)
    iou_list = []
    for cls in range(num_classes):
        pred_c   = (pred_labels == cls)
        target_c = (targets == cls)
        intersection = (pred_c & target_c).sum().float()
        union        = (pred_c | target_c).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        iou_list.append(iou)
    return iou_list, np.mean(iou_list)


def train_one_epoch(model, loader, criterion, optimizer, epoch, writer):
    """한 epoch 학습 후 TensorBoard에 Loss / mIoU 기록"""
    model.train()
    total_loss, total_miou, num_batches = 0.0, 0.0, 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)   # (B, 3, H, W)
        masks  = masks.to(device)    # (B, H, W)  Long

        # TODO 3: 학습 한 스텝을 완성하세요
        # 순서: zero_grad → forward → loss → backward → step
        #
        # 힌트:
        #   outputs = model(images)          # (B, C, H, W)
        #   loss    = criterion(outputs, masks)  # CrossEntropy는 (B,C,H,W)와 (B,H,W)를 받음

        optimizer.zero_grad()
        outputs = model(images)  # (B, C, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # ── 아래는 수정하지 마세요 ──
        with torch.no_grad():
            _, batch_miou = compute_iou(outputs.detach(), masks)
        total_loss  += loss.item()
        total_miou  += batch_miou
        num_batches += 1

    avg_loss = total_loss  / num_batches
    avg_miou = total_miou  / num_batches

    writer.add_scalar('UNet/Loss/train', avg_loss, epoch)
    writer.add_scalar('UNet/mIoU/train', avg_miou, epoch)
    return avg_loss, avg_miou


def evaluate(model, loader, criterion, epoch, writer, split='val'):
    """검증 / 테스트 평가 후 TensorBoard에 기록"""
    model.eval()
    total_loss, total_miou, num_batches = 0.0, 0.0, 0
    all_iou_per_class = [0.0] * PetSegDataset.NUM_CLASSES

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            # TODO 4: 평가 한 스텝을 완성하세요
            # 힌트: torch.no_grad() 블록 안에서 forward → loss 계산만 수행
            # (backward, optimizer.step 불필요)
            outputs = model(images)
            loss = criterion(outputs, masks)

            # ── 아래는 수정하지 마세요 ──
            iou_list, batch_miou = compute_iou(outputs, masks)
            total_loss  += loss.item()
            total_miou  += batch_miou
            num_batches += 1
            for c in range(PetSegDataset.NUM_CLASSES):
                all_iou_per_class[c] += iou_list[c]

    avg_loss = total_loss  / num_batches
    avg_miou = total_miou  / num_batches
    avg_iou_per_class = [v / num_batches for v in all_iou_per_class]

    writer.add_scalar(f'UNet/Loss/{split}', avg_loss, epoch)
    writer.add_scalar(f'UNet/mIoU/{split}', avg_miou, epoch)
    return avg_loss, avg_miou, avg_iou_per_class

# 옵티마이저 / 스케줄러 / 손실 함수 (제공)
# ignore_index=2: 'Not classified' 픽셀은 loss 계산에서 제외
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

run_id = time.strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir=f'runs/UNet_{run_id}')

# 모델 그래프 기록
dummy_input = torch.zeros(1, 3, 128, 128).to(device)
writer.add_graph(model, dummy_input)

EPOCHS = 30
history = {'train_loss':[], 'val_loss':[], 'train_miou':[], 'val_miou':[]}

print(f"{'='*65}")
print(f"  U-Net 학습 시작 | {EPOCHS} epochs | device: {device}")
print(f"{'='*65}")
print(f"{'Epoch':>6} | {'Tr Loss':>8} | {'Tr mIoU':>8} | {'Val Loss':>9} | {'Val mIoU':>9}")
print(f"{'-'*65}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_miou           = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
    val_loss, val_miou, val_iou_cls = evaluate(model, val_loader, criterion, epoch, writer, 'val')
    scheduler.step()

    history['train_loss'].append(tr_loss);  history['train_miou'].append(tr_miou)
    history['val_loss'].append(val_loss);   history['val_miou'].append(val_miou)

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6} | {tr_loss:>8.4f} | {tr_miou:>7.4f} | {val_loss:>9.4f} | {val_miou:>8.4f}")

print(f"\n최종 Val mIoU : {history['val_miou'][-1]:.4f}")
print(f"TensorBoard 로그: runs/UNet_{run_id}")
# TensorBoard에 예측 이미지 기록 (제공)
def log_prediction_images(model, loader, writer, epoch, n=4, tag='val_predictions'):
    model.eval()
    images, masks = next(iter(loader))
    images_dev = images[:n].to(device)

    with torch.no_grad():
        preds = model(images_dev).argmax(dim=1).cpu()  # (n, H, W)

    # 마스크 → RGB 변환
    def to_rgb(mask): return torch.tensor(LABEL_COLORS[mask.numpy()]).permute(2,0,1).float()

    grid_imgs, grid_preds, grid_gts = [], [], []
    for i in range(n):
        grid_imgs.append(images[i] * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1))
        grid_gts.append(to_rgb(masks[i]))
        grid_preds.append(to_rgb(preds[i]))

    import torchvision.utils as vutils
    writer.add_image(f'{tag}/input',   vutils.make_grid(grid_imgs,  nrow=n), epoch)
    writer.add_image(f'{tag}/gt_mask', vutils.make_grid(grid_gts,   nrow=n), epoch)
    writer.add_image(f'{tag}/pred',    vutils.make_grid(grid_preds, nrow=n), epoch)

log_prediction_images(model, val_loader, writer, epoch=EPOCHS)
writer.close()
print("TensorBoard IMAGES 탭에 예측 마스크가 기록되었습니다.")

# 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

epochs_range = range(1, EPOCHS + 1)
axes[0].plot(epochs_range, history['train_loss'], 'o-', label='Train', color='#e74c3c', linewidth=2)
axes[0].plot(epochs_range, history['val_loss'],   's-', label='Val',   color='#2980b9', linewidth=2)
axes[0].set_title('Loss 곡선', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history['train_miou'], 'o-', label='Train', color='#e74c3c', linewidth=2)
axes[1].plot(epochs_range, history['val_miou'],   's-', label='Val',   color='#2980b9', linewidth=2)
axes[1].set_title('mIoU 곡선', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mIoU')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('U-Net 학습 곡선 (Oxford-IIIT Pet)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# 테스트셋 최종 평가
test_loss, test_miou, test_iou_cls = evaluate(
    model, test_loader, criterion, epoch=0, writer=SummaryWriter('/tmp/dummy'), split='test'
)

print("\n" + "="*45)
print(f"  테스트셋 최종 결과")
print("="*45)
print(f"  Loss  : {test_loss:.4f}")
print(f"  mIoU  : {test_miou:.4f}")
print("-"*45)
for i, cls_name in enumerate(PetSegDataset.CLASS_NAMES):
    print(f"  {cls_name:<18}: IoU = {test_iou_cls[i]:.4f}")
print("="*45)

# 클래스별 IoU 바 차트
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#3498db','#bdc3c7','#e67e22']
bars = ax.bar(PetSegDataset.CLASS_NAMES, test_iou_cls, color=colors, width=0.5)
ax.set_ylim(0, 1); ax.set_ylabel('IoU'); ax.set_title('클래스별 IoU (테스트셋)', fontsize=12, fontweight='bold')
ax.axhline(y=test_miou, linestyle='--', color='red', label=f'mIoU = {test_miou:.4f}')
for bar, val in zip(bars, test_iou_cls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10)
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

# 예측 결과 시각화 (테스트셋)
model.eval()
images, masks = next(iter(test_loader))
images_dev = images[:6].to(device)

with torch.no_grad():
    preds = model(images_dev).argmax(dim=1).cpu()

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
row_labels = ["입력 이미지", "정답 마스크", "예측 마스크"]

for i in range(6):
    axes[0, i].imshow(denorm(images[i]));       axes[0, i].axis('off')
    axes[1, i].imshow(mask_to_rgb(masks[i]));   axes[1, i].axis('off')
    axes[2, i].imshow(mask_to_rgb(preds[i]));   axes[2, i].axis('off')

for ax, label in zip(axes[:, 0], row_labels):
    ax.set_ylabel(label, fontsize=11, fontweight='bold')

patches = [mpatches.Patch(color=LABEL_COLORS[i], label=PetSegDataset.CLASS_NAMES[i]) for i in range(3)]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10)
plt.suptitle("U-Net 예측 결과 (테스트셋)", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
