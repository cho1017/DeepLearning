import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 윈도우 기본 한글 폰트(맑은 고딕) 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
# 재현성 고정
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")
print(f"PyTorch 버전 : {torch.__version__}")

# 전처리 파이프라인
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # 좌우 반전 (데이터 증강)
    transforms.RandomCrop(32, padding=4),    # 랜덤 크롭 (데이터 증강)
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# 데이터셋 로드 (전체 사용)
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']
NUM_CLASSES = 10

print(f"훈련 데이터: {len(train_dataset):,}장 ({len(train_loader)} 배치)")
print(f"테스트 데이터: {len(test_dataset):,}장 ({len(test_loader)} 배치)")

# 샘플 이미지 시각화
def denormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    return (tensor * std + mean).clamp(0, 1)

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i, ax in enumerate(axes.flatten()):
    img = denormalize(images[i]).permute(1,2,0).numpy()
    ax.imshow(img); ax.set_title(CLASSES[labels[i]], fontsize=8); ax.axis('off')
plt.suptitle("CIFAR-10 샘플 이미지 (훈련셋)", fontsize=12)
plt.tight_layout(); plt.show()
print("이미지 shape:", images[0].shape)  # (3, 32, 32)


def train_one_epoch(model, loader, criterion, optimizer, epoch, writer, tag):
    """한 epoch 학습 후 TensorBoard에 Loss / Accuracy 기록"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    # TensorBoard 기록
    writer.add_scalar(f'{tag}/Loss/train',     avg_loss, epoch)
    writer.add_scalar(f'{tag}/Accuracy/train', accuracy, epoch)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, epoch, writer, tag):
    """테스트셋 평가 후 TensorBoard에 기록"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    # TensorBoard 기록
    writer.add_scalar(f'{tag}/Loss/test',     avg_loss, epoch)
    writer.add_scalar(f'{tag}/Accuracy/test', accuracy, epoch)

    return avg_loss, accuracy


def run_training(model, model_name, epochs=20):
    """학습 전체 루프 실행 (TensorBoard writer 포함)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=f'runs/{model_name}')

    # TensorBoard에 모델 그래프 추가
    dummy = torch.zeros(1, 3, 32, 32).to(device)
    writer.add_graph(model, dummy)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print(f"\n{'='*55}")
    print(f"  {model_name} 학습 시작 | {epochs} epochs | device: {device}")
    print(f"{'='*55}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print(f"{'-'*55}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, model_name)
        te_loss, te_acc = evaluate(model, test_loader, criterion, epoch, writer, model_name)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.2f}% | {te_loss:>9.4f} | {te_acc:>7.2f}%")

    writer.close()
    print(f"\n최종 테스트 정확도: {history['test_acc'][-1]:.2f}%")
    print(f"TensorBoard 로그 저장 위치: runs/{model_name}")
    return history

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()                     # (B, 3, 32, 32) → (B, 3072)
        self.fc1     = nn.Linear(3 * 32 * 32, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = self.fc3(x)
        return x

mlp_model = MLP(NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
print(f"MLP 파라미터 수: {total_params:,}")
print(mlp_model)

# MLP 학습 (전체 데이터, 20 epoch)
mlp_history = run_training(mlp_model, model_name='MLP', epochs=20)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # ── TODO 1: Block 1 구현 ─────────────────────────────
        # Conv2d(3, 32, 3, padding=1) → BatchNorm2d(32) → ReLU
        # Conv2d(32, 32, 3, padding=1) → BatchNorm2d(32) → ReLU
        # MaxPool2d(2, 2) → Dropout(0.25)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )



        # ── TODO 2: Block 2 구현 ─────────────────────────────
        # Conv2d(32, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
        # Conv2d(64, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
        # MaxPool2d(2, 2) → Dropout(0.25)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # ── TODO 3: Block 3 구현 ─────────────────────────────
        # Conv2d(64, 128, 3, padding=1) → BatchNorm2d(128) → ReLU
        # Conv2d(128, 128, 3, padding=1) → BatchNorm2d(128) → ReLU
        # MaxPool2d(2, 2) → Dropout(0.25)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(64,128,1),
            nn.BatchNorm2d(128)

        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25)
        )



        # ── TODO 4: Classifier 구현 ──────────────────────────
        # AdaptiveAvgPool 후 128차원 → Linear(128, 256) → ReLU → Dropout(0.5) → Linear(256, num_classes)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))  # (B,128,4,4) → (B,128,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)

            # TODO: 여기를 채우세요
        )

    def forward(self, x):
        # TODO 5: forward 흐름 작성
        # block1 → block2 → block3 → avgpool → flatten → classifier
        x = self.block1(x)
        x = self.block2(x)
        shortcut = self.shortcut(x)
        x = self.block3(x) + shortcut
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
        # TODO: 수정하세요


# ── 모델 검증 ──────────────────────────────────────────────────
cnn_model = CNN(NUM_CLASSES).to(device)

dummy = torch.zeros(4, 3, 32, 32).to(device)
out   = cnn_model(dummy)
print(f"입력 shape : {dummy.shape}")
print(f"출력 shape : {out.shape}")    # 기대값: (4, 10)
assert out.shape == (4, 10), "❌ 출력 shape 오류! forward를 다시 확인하세요."
print("✅ shape 검증 통과")

total_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f"\nCNN 파라미터 수 : {total_params:,}")
print(f"MLP 파라미터 수 : {sum(p.numel() for p in mlp_model.parameters() if p.requires_grad):,}")

# CNN 학습 (전체 데이터, 20 epoch)
# 학습 중 TensorBoard를 새로고침하면 MLP와 CNN 곡선을 함께 볼 수 있습니다.
cnn_history = run_training(cnn_model, model_name='CNN', epochs=20)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = range(1, 21)
configs = [
    (0, 0, 'train_loss', 'Train Loss',     'Loss'),
    (0, 1, 'test_loss',  'Test `Loss',      'Loss'),
    (1, 0, 'train_acc',  'Train Accuracy', 'Accuracy (%)'),
    (1, 1, 'test_acc',   'Test Accuracy',  'Accuracy (%)'),
]

for r, c, key, title, ylabel in configs:
    ax = axes[r, c]
    ax.plot(epochs, mlp_history[key], 'o-', label='MLP',  color='#e74c3c', linewidth=2)
    ax.plot(epochs, cnn_history[key], 's-', label='CNN',  color='#2980b9', linewidth=2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("MLP vs CNN — CIFAR-10 학습 곡선 비교", fontsize=15, fontweight='bold')
plt.tight_layout(); plt.show()

# 최종 결과 요약
print("\n" + "="*45)
print(f"{'모델':<8} | {'Train Acc':>10} | {'Test Acc':>9}")
print("-"*45)
print(f"{'MLP':<8} | {mlp_history['train_acc'][-1]:>9.2f}% | {mlp_history['test_acc'][-1]:>8.2f}%")
print(f"{'CNN':<8} | {cnn_history['train_acc'][-1]:>9.2f}% | {cnn_history['test_acc'][-1]:>8.2f}%")
print("="*45)

# 클래스별 정확도 분석 (CNN)
cnn_model.eval()
class_correct = [0] * NUM_CLASSES
class_total   = [0] * NUM_CLASSES

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs  = cnn_model(images)
        _, preds = outputs.max(1)
        for label, pred in zip(labels, preds):
            class_correct[label] += (label == pred).item()
            class_total[label]   += 1

accs = [100 * class_correct[i] / class_total[i] for i in range(NUM_CLASSES)]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(CLASSES, accs, color=plt.cm.RdYlGn([a/100 for a in accs]))
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("CNN — 클래스별 테스트 정확도", fontsize=13, fontweight='bold')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout(); plt.show()

