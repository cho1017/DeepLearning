# 학습 결과 정리 (TensorBoard)

각 태스크별로 학습이 끝난 뒤 TensorBoard(`localhost:6006`)에 기록된 최종 지표를 정리한 문서입니다.
스크린샷 원본은 [`screenshots/`](./screenshots) 폴더에 있습니다.

## 1. Task1 — Image Classification

| 모델 | Epoch | Accuracy/train | Accuracy/test | Loss/train | Loss/test |
|---|---|---|---|---|---|
| MLP | 20 | 49.23% | 50.94% | 1.4193 | 1.4198 |
| CNN | 19 | 79.53% | 81.49% | 0.5936 | 0.5331 |

→ CNN이 MLP 대비 정확도 약 30%p 이상 높음. 이미지 분류에서 convolution 기반 특징 추출이 fully-connected 구조보다 훨씬 효과적이라는 걸 보여줌.

## 2. Task2 — Image Segmentation

| 모델 | Epoch | mIoU/train | mIoU/val | Loss/train | Loss/val |
|---|---|---|---|---|---|
| UNet | 7 | 0.6611 | 0.6478 | 0.3542 | 0.3915 |

→ 다른 모델들은 30 epoch까지 돌았는데 UNet만 7 epoch에서 멈춘 것으로 보임 (필요하면 early stopping 조건이나 epoch 수를 다시 확인해볼 것).

## 3. Task3 — Image Denoising

| 모델 | Epoch | PSNR/train | PSNR/test | SSIM/test | Loss/train | Loss/test |
|---|---|---|---|---|---|---|
| Autoencoder (Baseline) | 30 | 24.32 | 24.44 | 0.8165 | 0.0039 | 0.0038 |
| ImprovedDenoiser | 30 | 25.01 | 25.10 | 0.8403 | 0.0033 | 0.0033 |

→ Improved 버전이 PSNR +0.66dB, SSIM +0.024 개선. 구조 개선(잔차 연결 등)이 노이즈 제거 품질을 확실히 끌어올림.

## 4. Task4 — Image Colorization

| 모델 | Epoch | PSNR/train | PSNR/val | SSIM/val | Loss/train | Loss/val |
|---|---|---|---|---|---|---|
| Baseline | 30 | 23.54 | 23.53 | 0.9205 | 0.0144 | 0.0140 |
| Improved (Skip + Residual + Attention Gate) | 30 | 23.61 | 23.62 | 0.9232 | 0.0141 | 0.0138 |

→ Attention Gate를 추가한 Improved 모델이 PSNR/SSIM 모두 소폭이지만 일관되게 Baseline을 앞섬. 정성적 결과(`screenshots/01_images_tab_qualitative_results.png`)에서도 gt_rgb 대비 pred_rgb 색상 복원이 자연스러움.

## 스크린샷 목록

| 파일 | 내용 |
|---|---|
| `01_images_tab_qualitative_results.png` | IMAGES 탭 — Autoencoder/Baseline/Improved/ImprovedDenoiser/UNet(val_predictions) 정성적 결과 비교 |
| `02_scalars_autoencoder_baseline_cnn.png` | SCALARS — Autoencoder, Baseline, CNN 곡선 |
| `03_scalars_baseline_cnn_improved.png` | SCALARS — Baseline, CNN, Improved 곡선 |
| `04_scalars_cnn_improved_improveddenoiser.png` | SCALARS — CNN, Improved, ImprovedDenoiser 곡선 |
| `05_scalars_improveddenoiser_mlp_unet.png` | SCALARS — ImprovedDenoiser, MLP, UNet 곡선 |
| `06_timeseries_autoencoder_baseline.png` | TIME SERIES — Autoencoder, Baseline 카드 |
| `07_timeseries_cnn_improved.png` | TIME SERIES — CNN, Improved 카드 |
| `08_timeseries_improved_improveddenoiser.png` | TIME SERIES — Improved, ImprovedDenoiser 카드 |
| `09_timeseries_improveddenoiser_full.png` | TIME SERIES — ImprovedDenoiser 전체 (clean/noisy/restored 이미지 포함) |
| `10_timeseries_baseline_full.png` | TIME SERIES — Baseline 전체 곡선 |
| `11_timeseries_mlp_unet_valpredictions.png` | TIME SERIES — MLP, UNet, val_predictions 카드 |

*(수치는 TensorBoard 스크린샷의 Smoothing=0.6 기준 "Value" 열 — 마지막 step 실측값입니다.)*
