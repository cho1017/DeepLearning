import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sympy.printing import pytorch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim
from thop import profile as thop_profile
import random, copy, time

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스 : {device}")
print(f"PyTorch 버전  : {torch.__version__}")

# CIFAR-10 다운로드 (정규화 없이 — 노이즈 추가 후 정규화)
_base_transform = transforms.ToTensor()

_train_raw = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True,  transform=_base_transform)
_test_raw  = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=_base_transform)

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR10_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

# 노이즈 강도별 시각화
fig, axes = plt.subplots(4, 6, figsize=(14, 10))
sigmas = [0.0, 0.1, 0.2, 0.3]
sample_img, _ = _train_raw[42]     # (3, 32, 32) float [0,1]

for row, sigma in enumerate(sigmas):
    for col in range(6):
        noisy = (sample_img + torch.randn_like(sample_img) * sigma).clamp(0, 1)
        axes[row, col].imshow(noisy.permute(1,2,0).numpy())
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(f'σ = {sigma}', fontsize=11, fontweight='bold')

plt.suptitle("가우시안 노이즈 강도별 샘플 (σ = 0.0 / 0.1 / 0.2 / 0.3)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# sigma별 PSNR 분석
print(f"{'sigma':>6} | {'PSNR (dB)':>10} | 품질 판단")
print("-" * 35)
for sigma in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
    psnr_vals = []
    for i in range(50):
        orig, _ = _train_raw[i]
        noisy   = (orig + torch.randn_like(orig) * sigma).clamp(0,1)
        psnr_vals.append(calc_psnr(orig.numpy(), noisy.numpy(), data_range=1.0))
    avg = np.mean(psnr_vals)
    quality = "양호" if avg >= 30 else ("보통" if avg >= 25 else "불량")
    print(f"{sigma:>6.2f} | {avg:>10.2f} | {quality}")

class DenoisingDataset(Dataset):
    """
    CIFAR-10 기반 가우시안 노이즈 Denoising Dataset

    Args:
        base_dataset : torchvision CIFAR10 인스턴스 (transform=ToTensor())
        sigma        : 가우시안 노이즈 표준편차
        augment      : 랜덤 수평 반전 여부 (훈련용)
    """
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

    def __init__(self, base_dataset, sigma=0.2, augment=False):
        self.dataset = base_dataset
        self.sigma   = sigma
        self.augment = augment

    def __len__(self):
        # TODO 1: 전체 데이터 수를 반환하세요
        return len(self.dataset)
        pass

    def __getitem__(self, idx):
        # TODO 2: (noisy_normalized, clean_normalized) 쌍을 반환하세요
        #
        # 처리 순서:
        #   (1) self.dataset[idx] 로 (PIL→Tensor) 이미지를 가져온다  ← label은 버림
        #   (2) augment=True일 때 50% 확률로 좌우 반전
        #       힌트: if random.random() > 0.5: img = torch.flip(img, dims=[2])
        #   (3) 가우시안 노이즈 생성 및 추가
        #       noise    = torch.randn_like(img) * self.sigma
        #       noisy    = (img + noise).clamp(0, 1)
        #   (4) 이미지와 노이즈 이미지를 각각 정규화
        #       clean_norm = (img   - self.MEAN) / self.STD
        #       noisy_norm = (noisy - self.MEAN) / self.STD
        #   (5) (noisy_norm, clean_norm) 반환  ← 입력: 노이즈, 타깃: 원본
        img, _ = self.dataset[idx]
        if self.augment and random.random() < 0.5:
            img = torch.flip(img, [2])
        noise = torch.randn_like(img)*self.sigma
        noisy = (img + noise).clamp(0,1)

        clean_norm = (img - self.MEAN) / self.STD
        noisy_norm = (noisy - self.MEAN) / self.STD

        return noisy_norm, clean_norm

# ── DataLoader 생성 ───────────────────────────────────────────
SIGMA = 0.2

train_dataset = DenoisingDataset(_train_raw, sigma=SIGMA, augment=True)
test_dataset  = DenoisingDataset(_test_raw,  sigma=SIGMA, augment=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
print(f"Train : {len(train_dataset):,}장  ({len(train_loader)} 배치)")
print(f"Test  : {len(test_dataset):,}장  ({len(test_loader)} 배치)")

# ── Dataset 검증 ─────────────────────────────────────────────
noisy_t, clean_t = train_dataset[0]

print("[shape 검증]")
print(f"  noisy shape : {noisy_t.shape}")   # 기대: (3, 32, 32)
print(f"  clean shape : {clean_t.shape}")   # 기대: (3, 32, 32)
print(f"  dtype       : {noisy_t.dtype}")   # 기대: torch.float32

assert noisy_t.shape == (3, 32, 32), "❌ noisy shape 오류"
assert clean_t.shape == (3, 32, 32), "❌ clean shape 오류"
assert noisy_t.dtype == torch.float32, "❌ dtype은 float32 이어야 합니다"
assert not torch.allclose(noisy_t, clean_t), "❌ noisy와 clean이 동일합니다 — 노이즈가 추가되지 않았습니다"
print("✅ 모든 shape 검증 통과")

# ── 시각화 ────────────────────────────────────────────────────
def denorm(t):
    return (t * DenoisingDataset.STD + DenoisingDataset.MEAN).clamp(0,1).permute(1,2,0).numpy()

fig, axes = plt.subplots(3, 8, figsize=(16, 7))
for i in range(8):
    n, c = train_dataset[i * 300]
    n_img, c_img = denorm(n), denorm(c)
    psnr_val = calc_psnr(c_img, n_img, data_range=1.0)

    axes[0,i].imshow(c_img);  axes[0,i].set_title("원본", fontsize=8);       axes[0,i].axis('off')
    axes[1,i].imshow(n_img);  axes[1,i].set_title("노이즈", fontsize=8);     axes[1,i].axis('off')
    axes[2,i].imshow(np.abs(c_img - n_img))
    axes[2,i].set_title(f"차이\nPSNR:{psnr_val:.1f}dB", fontsize=7)
    axes[2,i].axis('off')

for ax, label in zip(axes[:,0], ["원본(타깃)", f"노이즈(σ={SIGMA})", "|차이|"]):
    ax.set_ylabel(label, fontsize=10, fontweight='bold')

plt.suptitle(f"CIFAR-10 Denoising Dataset 샘플 (σ={SIGMA})", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

class Autoencoder(nn.Module):
    """베이스라인 Convolutional Autoencoder"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                    # 32→16
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                    # 16→8
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32, 2, stride=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.bottleneck(self.encoder(x)))


ae_model = Autoencoder().to(device)
dummy    = torch.zeros(2, 3, 32, 32).to(device)
assert ae_model(dummy).shape == (2, 3, 32, 32), "❌ Autoencoder shape 오류"

ae_params = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
print(f"Autoencoder 파라미터 수 : {ae_params:,}")
print("✅ Autoencoder shape 검증 통과")

class ImprovedDenoiser(nn.Module):
    """
    개선된 Denoising 모델 — 직접 설계하세요

    조건:
      - 입력 : (B, 3, 32, 32)
      - 출력 : (B, 3, 32, 32), 값 범위 [0, 1]
      - 베이스라인 Autoencoder보다 높은 PSNR을 목표로 함

    힌트 (아래 중 하나 이상 적용):
      A) Skip Connection
         enc1 = self.enc1(x)                        # (B, 32, 32, 32)
         enc2 = self.enc2(pool(enc1))               # (B, 64, 16, 16)
         b    = self.bottleneck(pool(enc2))          # (B,128,  8,  8)
         d1   = self.dec1(cat([up1(b),  enc2], 1))  # (B, 64, 16, 16)
         d2   = self.dec2(cat([up2(d1), enc1], 1))  # (B, 32, 32, 32)
         out  = sigmoid(self.out(d2))

      B) Residual Block
         class ResBlock(nn.Module):
             def forward(self, x): return x + self.conv(x)

      C) DnCNN 스타일 (잔차 학습)
         noise_pred = self.net(noisy)   # 노이즈 예측
         return (noisy - noise_pred).clamp(0, 1)
    """
    def __init__(self):
        super().__init__()
        # TODO 3: 모델 구조를 정의하세요

        def block(inp, outp):
            return nn.Sequential(
                nn.Conv2d(inp, outp, 3, padding=1, bias=False),
                nn.BatchNorm2d(outp),
                nn.ReLU(inplace=True),
            )

        def block2(inp, outp):
            return nn.Sequential(
                nn.ConvTranspose2d(inp, outp, 2, stride=2, bias=False),
                nn.BatchNorm2d(outp),
                nn.ReLU(inplace=True),
            )
        self.enc1 = block(3, 32)
        self.enc2 = block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(64, 128)

        self.up1 = block2(128, 64)
        self.up2 = block2(64, 32)
        self.dec1 = block(128, 64)
        self.dec2 = block(64, 32)
        self.out = nn.Conv2d(32, 3, 3, padding=1)
        # (레이어 이름과 채널 수는 자유롭게 설계)

    def forward(self, x):
        # TODO 4: forward 흐름을 구현하세요
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        b = self.bottleneck(p2)
        up1 = self.up1(b)
        d1 = self.dec1(torch.cat([up1, e2], 1))
        up2 = self.up2(d1)
        d2 = self.dec2(torch.cat([up2, e1], 1))
        out1 = self.out(d2)
        return out1.sigmoid()

        # 반드시 (B, 3, 32, 32) shape을 반환해야 합니다

class ImprovedDenoiser2(nn.Module):
    """
    개선된 Denoising 모델 — 직접 설계하세요

    조건:
      - 입력 : (B, 3, 32, 32)
      - 출력 : (B, 3, 32, 32), 값 범위 [0, 1]
      - 베이스라인 Autoencoder보다 높은 PSNR을 목표로 함

    힌트 (아래 중 하나 이상 적용):
      A) Skip Connection
         enc1 = self.enc1(x)                        # (B, 32, 32, 32)
         enc2 = self.enc2(pool(enc1))               # (B, 64, 16, 16)
         b    = self.bottleneck(pool(enc2))          # (B,128,  8,  8)
         d1   = self.dec1(cat([up1(b),  enc2], 1))  # (B, 64, 16, 16)
         d2   = self.dec2(cat([up2(d1), enc1], 1))  # (B, 32, 32, 32)
         out  = sigmoid(self.out(d2))

      B) Residual Block
         class ResBlock(nn.Module):
             def forward(self, x): return x + self.conv(x)

      C) DnCNN 스타일 (잔차 학습)
         noise_pred = self.net(noisy)   # 노이즈 예측
         return (noisy - noise_pred).clamp(0, 1)
    """
    def __init__(self):
        super().__init__()
        # TODO 3: 모델 구조를 정의하세요

    def forward(self, x):  # x: 정규화된 noisy
        noise_pred = self.net(x)  # 노이즈 예측 (정규화 스케일)
        x01 = x * STD_t + MEAN_t  # 입력을 0~1로 복원
        return (x01 - noise_pred).clamp(0, 1)  # 학습 타깃(0~1)과 범위가 맞음

class ImprovedDenoiser3(nn.Module):
    """
    개선된 Denoising 모델 — 직접 설계하세요

    조건:
      - 입력 : (B, 3, 32, 32)
      - 출력 : (B, 3, 32, 32), 값 범위 [0, 1]
      - 베이스라인 Autoencoder보다 높은 PSNR을 목표로 함

    힌트 (아래 중 하나 이상 적용):
      A) Skip Connection
         enc1 = self.enc1(x)                        # (B, 32, 32, 32)
         enc2 = self.enc2(pool(enc1))               # (B, 64, 16, 16)
         b    = self.bottleneck(pool(enc2))          # (B,128,  8,  8)
         d1   = self.dec1(cat([up1(b),  enc2], 1))  # (B, 64, 16, 16)
         d2   = self.dec2(cat([up2(d1), enc1], 1))  # (B, 32, 32, 32)
         out  = sigmoid(self.out(d2))

      B) Residual Block
         class ResBlock(nn.Module):
             def forward(self, x): return x + self.conv(x)

      C) DnCNN 스타일 (잔차 학습)
         noise_pred = self.net(noisy)   # 노이즈 예측
         return (noisy - noise_pred).clamp(0, 1)
    """
    def __init__(self):
        super().__init__()
        # TODO 3: 모델 구조를 정의하세요

    def forward(self, x):  # x: 정규화된 noisy
        noise_pred = self.net(x)  # 노이즈 예측 (정규화 스케일)
        x01 = x * STD_t + MEAN_t  # 입력을 0~1로 복원
        return (x01 - noise_pred).clamp(0, 1)  # 학습 타깃(0~1)과 범위가 맞음

# ── shape 검증 ────────────────────────────────────────────────
improved_model = ImprovedDenoiser().to(device)
dummy = torch.zeros(2, 3, 32, 32).to(device)
out   = improved_model(dummy)
print(f"입력  shape : {dummy.shape}")
print(f"출력  shape : {out.shape}")     # 기대: (2, 3, 32, 32)
assert out.shape == (2, 3, 32, 32), "❌ 출력 shape 오류 — forward를 확인하세요"
assert out.min() >= 0 and out.max() <= 1, "❌ 출력 값이 [0,1] 범위를 벗어났습니다"
print("✅ ImprovedDenoiser shape / 값 범위 검증 통과")

imp_params = sum(p.numel() for p in improved_model.parameters() if p.requires_grad)
print(f"\nImprovedDenoiser 파라미터 수 : {imp_params:,}")
print(f"Autoencoder     파라미터 수 : {ae_params:,}")

MEAN_t = DenoisingDataset.MEAN.to(device)
STD_t  = DenoisingDataset.STD.to(device)

def denorm_batch(t):
    """(B,3,H,W) 정규화 텐서 → [0,1] 픽셀 값으로 복원"""
    return (t * STD_t + MEAN_t).clamp(0, 1)

def batch_psnr(pred, target):
    """배치 평균 PSNR 계산 (텐서, [0,1])"""
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1,2,3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, epoch, writer, tag):
    """한 epoch 학습 후 TensorBoard에 Loss / PSNR 기록"""
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        target = denorm_batch(clean)   # [0,1]로 복원

        # TODO 5: 학습 한 스텝을 완성하세요
        # 순서: zero_grad → forward(noisy) → loss(output, target) → backward → step
        # output = model(noisy)   # (B,3,32,32) in [0,1]
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_psnr += batch_psnr(output.detach(), target)
        total_loss += loss.item()
        n += 1

    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    writer.add_scalar(f'{tag}/Loss/train', avg_loss, epoch)
    writer.add_scalar(f'{tag}/PSNR/train', avg_psnr, epoch)
    return avg_loss, avg_psnr


def evaluate(model, loader, criterion, epoch, writer, tag):
    """테스트셋 평가 후 TensorBoard에 기록"""
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    all_ssim = []

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            target = denorm_batch(clean)

            # TODO 6: 평가 한 스텝을 완성하세요
            # (backward, optimizer.step 불필요)
            output = model(noisy)
            loss = criterion(output, target)
            pass  # TODO: 이 줄을 지우고 구현하세요

            total_loss += loss.item()
            total_psnr += batch_psnr(output, target)
            n += 1

            # SSIM 계산 (배치 샘플 4장)
            for k in range(min(4, output.shape[0])):
                pred_np   = output[k].cpu().permute(1,2,0).numpy()
                target_np = target[k].cpu().permute(1,2,0).numpy()
                all_ssim.append(calc_ssim(target_np, pred_np,
                                          data_range=1.0, channel_axis=2))

    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    avg_ssim = np.mean(all_ssim)
    writer.add_scalar(f'{tag}/Loss/test', avg_loss, epoch)
    writer.add_scalar(f'{tag}/PSNR/test', avg_psnr, epoch)
    writer.add_scalar(f'{tag}/SSIM/test', avg_ssim, epoch)
    return avg_loss, avg_psnr, avg_ssim

def run_training(model, tag, epochs=30, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer    = SummaryWriter(log_dir=f'runs/{tag}')

    # 모델 그래프 기록
    writer.add_graph(model, torch.zeros(1,3,32,32).to(device))

    history = {'train_loss':[], 'test_loss':[], 'train_psnr':[], 'test_psnr':[], 'test_ssim':[]}

    print(f"\n{'='*68}")
    print(f"  {tag} 학습 시작 | {epochs} epochs | device: {device}")
    print(f"{'='*68}")
    print(f"{'Epoch':>6} | {'TrLoss':>7} | {'TrPSNR':>7} | {'TeLoss':>7} | {'TePSNR':>7} | {'TeSSIM':>7}")
    print(f"{'-'*68}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_psnr            = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, tag)
        te_loss, te_psnr, te_ssim   = evaluate(model, test_loader, criterion, epoch, writer, tag)
        scheduler.step()

        history['train_loss'].append(tr_loss); history['train_psnr'].append(tr_psnr)
        history['test_loss'].append(te_loss);  history['test_psnr'].append(te_psnr)
        history['test_ssim'].append(te_ssim)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {tr_loss:>7.4f} | {tr_psnr:>7.2f} | {te_loss:>7.4f} | {te_psnr:>7.2f} | {te_ssim:>7.4f}")

    elapsed = time.time() - start
    writer.close()
    print(f"\n학습 완료 ({elapsed:.0f}초) | 최종 Test PSNR: {history['test_psnr'][-1]:.2f} dB  SSIM: {history['test_ssim'][-1]:.4f}")
    return history

# ── Autoencoder 학습 ─────────────────────────────────────────
ae_history = run_training(ae_model, tag='Autoencoder', epochs=30)

# ── ImprovedDenoiser 학습 ────────────────────────────────────
imp_history = run_training(improved_model, tag='ImprovedDenoiser', epochs=30)

# TensorBoard에 복원 이미지 기록
def log_denoising_images(model, loader, writer, tag, n=8):
    model.eval()
    noisy_batch, clean_batch = next(iter(loader))
    noisy_dev = noisy_batch[:n].to(device)

    with torch.no_grad():
        restored = model(noisy_dev).cpu()

    noisy_vis   = denorm_batch(noisy_batch[:n].to(device)).cpu()
    clean_vis   = denorm_batch(clean_batch[:n].to(device)).cpu()

    writer_tmp = SummaryWriter(log_dir=f'runs/{tag}')
    writer_tmp.add_image(f'{tag}/noisy',     vutils.make_grid(noisy_vis,   nrow=n), 0)
    writer_tmp.add_image(f'{tag}/clean',     vutils.make_grid(clean_vis,   nrow=n), 0)
    writer_tmp.add_image(f'{tag}/restored',  vutils.make_grid(restored,    nrow=n), 0)
    writer_tmp.close()
    print(f"TensorBoard IMAGES 탭에 [{tag}] 복원 이미지가 기록되었습니다.")

log_denoising_images(ae_model,       train_loader, None, 'Autoencoder')
log_denoising_images(improved_model, train_loader, None, 'ImprovedDenoiser')

# ── FLOPs & Params 측정 (thop 라이브러리) ────────────────────
def get_model_stats(model, input_size=(1, 3, 32, 32)):
    """FLOPs, Params, 추론 시간(ms) 반환"""
    dummy = torch.zeros(input_size).to(device)
    model.eval()

    # FLOPs & Params
    flops, params = thop_profile(model, inputs=(dummy,), verbose=False)

    # 추론 시간 (GPU warm-up 포함, 100회 평균)
    with torch.no_grad():
        for _ in range(10): model(dummy)           # warm-up
        start = time.time()
        for _ in range(100): model(dummy)
        latency_ms = (time.time() - start) / 100 * 1000

    return flops, params, latency_ms

ae_flops,  ae_params_cnt,  ae_lat  = get_model_stats(ae_model)
imp_flops, imp_params_cnt, imp_lat = get_model_stats(improved_model)

print(f"{'지표':<20} | {'Autoencoder':>15} | {'ImprovedDenoiser':>18}")
print("-" * 60)
print(f"{'Params':<20} | {ae_params_cnt:>15,.0f} | {imp_params_cnt:>18,.0f}")
print(f"{'FLOPs':<20} | {ae_flops:>15,.0f} | {imp_flops:>18,.0f}")
print(f"{'FLOPs (GMac)':<20} | {ae_flops/1e9:>14.3f}G | {imp_flops/1e9:>17.3f}G")
print(f"{'Latency (ms)':<20} | {ae_lat:>14.3f}  | {imp_lat:>17.3f}")

# ── 최종 성능 비교 ───────────────────────────────────────────
ae_final_psnr  = ae_history['test_psnr'][-1]
ae_final_ssim  = ae_history['test_ssim'][-1]
imp_final_psnr = imp_history['test_psnr'][-1]
imp_final_ssim = imp_history['test_ssim'][-1]

# 노이즈 이미지 자체의 PSNR (모델 없이)
noisy_psnr_list = []
for i in range(200):
    n, c = test_dataset[i]
    n_img = denorm_batch(n.unsqueeze(0).to(device))[0].cpu().permute(1,2,0).numpy()
    c_img = denorm_batch(c.unsqueeze(0).to(device))[0].cpu().permute(1,2,0).numpy()
    noisy_psnr_list.append(calc_psnr(c_img, n_img, data_range=1.0))
noisy_baseline_psnr = np.mean(noisy_psnr_list)

print("\n" + "="*65)
print(f"  {'모델':<22} | {'PSNR (dB)':>10} | {'SSIM':>7} | {'Params':>10} | {'FLOPs':>10}")
print("-"*65)
print(f"  {'노이즈 입력 (복원 없음)':<22} | {noisy_baseline_psnr:>10.2f} |  {'N/A':>5} | {'N/A':>10} | {'N/A':>10}")
print(f"  {'Autoencoder':<22} | {ae_final_psnr:>10.2f} | {ae_final_ssim:>7.4f} | {ae_params_cnt:>10,.0f} | {ae_flops/1e6:>8.1f}M")
print(f"  {'ImprovedDenoiser':<22} | {imp_final_psnr:>10.2f} | {imp_final_ssim:>7.4f} | {imp_params_cnt:>10,.0f} | {imp_flops/1e6:>8.1f}M")
print("="*65)
psnr_gain = imp_final_psnr - ae_final_psnr
print(f"\n  ImprovedDenoiser PSNR 향상 : {psnr_gain:+.2f} dB  ({'개선' if psnr_gain > 0 else '미달성 — 모델을 다시 설계해보세요'})")

# ── 종합 비교 차트 ───────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

epochs_r = range(1, 31)

# 1) Loss 곡선
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_r, ae_history['test_loss'],  'o-', label='Autoencoder',     color='#e74c3c', lw=2)
ax1.plot(epochs_r, imp_history['test_loss'], 's-', label='ImprovedDenoiser',color='#2980b9', lw=2)
ax1.set_title('Test Loss (MSE)', fontweight='bold'); ax1.set_xlabel('Epoch')
ax1.legend(); ax1.grid(True, alpha=0.3)

# 2) PSNR 곡선
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(y=noisy_baseline_psnr, linestyle=':', color='gray', label=f'노이즈 입력 ({noisy_baseline_psnr:.1f}dB)')
ax2.plot(epochs_r, ae_history['test_psnr'],  'o-', label='Autoencoder',     color='#e74c3c', lw=2)
ax2.plot(epochs_r, imp_history['test_psnr'], 's-', label='ImprovedDenoiser',color='#2980b9', lw=2)
ax2.set_title('Test PSNR (dB) ↑', fontweight='bold'); ax2.set_xlabel('Epoch')
ax2.legend(); ax2.grid(True, alpha=0.3)

# 3) SSIM 곡선
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs_r, ae_history['test_ssim'],  'o-', label='Autoencoder',     color='#e74c3c', lw=2)
ax3.plot(epochs_r, imp_history['test_ssim'], 's-', label='ImprovedDenoiser',color='#2980b9', lw=2)
ax3.set_title('Test SSIM ↑', fontweight='bold'); ax3.set_xlabel('Epoch')
ax3.legend(); ax3.grid(True, alpha=0.3)

# 4) Params 비교
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar(['Autoencoder','ImprovedDenoiser'], [ae_params_cnt/1e6, imp_params_cnt/1e6],
               color=['#e74c3c','#2980b9'], width=0.5)
ax4.set_title('Parameters (M) ↓', fontweight='bold'); ax4.set_ylabel('Millions')
for b, v in zip(bars, [ae_params_cnt/1e6, imp_params_cnt/1e6]):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.3f}M', ha='center', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 5) FLOPs 비교
ax5 = fig.add_subplot(gs[1, 1])
bars2 = ax5.bar(['Autoencoder','ImprovedDenoiser'], [ae_flops/1e9, imp_flops/1e9],
                color=['#e74c3c','#2980b9'], width=0.5)
ax5.set_title('FLOPs (GMACs) ↓', fontweight='bold'); ax5.set_ylabel('GMACs')
for b, v in zip(bars2, [ae_flops/1e9, imp_flops/1e9]):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.0002, f'{v:.3f}G', ha='center', fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# 6) PSNR vs Params 산점도 (트레이드오프)
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter([ae_params_cnt/1e6],  [ae_final_psnr],  s=200, color='#e74c3c', zorder=5, label='Autoencoder')
ax6.scatter([imp_params_cnt/1e6], [imp_final_psnr], s=200, color='#2980b9', zorder=5, label='ImprovedDenoiser')
ax6.set_xlabel('Parameters (M)'); ax6.set_ylabel('PSNR (dB)')
ax6.set_title('PSNR vs Params (트레이드오프)', fontweight='bold')
ax6.legend(); ax6.grid(True, alpha=0.3)

plt.suptitle('Autoencoder vs ImprovedDenoiser — 종합 비교', fontsize=14, fontweight='bold')
plt.show()

# 복원 이미지 정성 비교
ae_model.eval(); improved_model.eval()
noisy_batch, clean_batch = next(iter(test_loader))
noisy_dev = noisy_batch[:6].to(device)

with torch.no_grad():
    ae_out  = ae_model(noisy_dev).cpu()
    imp_out = improved_model(noisy_dev).cpu()

noisy_vis = denorm_batch(noisy_batch[:6].to(device)).cpu()
clean_vis = denorm_batch(clean_batch[:6].to(device)).cpu()

def to_np(t): return t.permute(1,2,0).numpy().clip(0,1)

fig, axes = plt.subplots(4, 6, figsize=(18, 12))
row_labels = ["노이즈 입력", "Autoencoder 복원", "ImprovedDenoiser 복원", "원본(정답)"]

for col in range(6):
    axes[0,col].imshow(to_np(noisy_vis[col]));  axes[0,col].axis('off')
    axes[1,col].imshow(to_np(ae_out[col]));     axes[1,col].axis('off')
    axes[2,col].imshow(to_np(imp_out[col]));    axes[2,col].axis('off')
    axes[3,col].imshow(to_np(clean_vis[col]));  axes[3,col].axis('off')

    # PSNR 주석
    for row_idx, pred in enumerate([noisy_vis[col], ae_out[col], imp_out[col]]):
        p = calc_psnr(to_np(clean_vis[col]), to_np(pred), data_range=1.0)
        axes[row_idx, col].set_title(f"PSNR:{p:.1f}dB", fontsize=8)

for ax, label in zip(axes[:,0], row_labels):
    ax.set_ylabel(label, fontsize=10, fontweight='bold')

plt.suptitle(f"복원 결과 정성 비교 (σ={SIGMA})", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
