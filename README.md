# MM_Fusion_OGMGE

**OGM-GE on AV-MNIST - convergence study (PAI-SG Multimodal Wave 1)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the multimodal convergence comparison in [MM_Wave1A_AVMNIST](https://github.com/Statera-Workshop/MM_Wave1A_AVMNIST).

## Method

Linear concat head; per-batch modality scores from the split head, coefficient 1 - tanh(alpha x relu(ratio)) on the dominant encoder's conv gradients (official code form), alpha 0.1, modulation epochs 0-50. *OGM-GE* adds Gaussian noise with the gradient's std to the conv gradients of **both** encoders (official `OGM_GE`); *OGM* scales only (official `OGM`). *SGD (official)*: SGD 1e-3, momentum 0.9, wd 1e-4, StepLR(70, 0.1).

**Common setup** (identical for the AV-MNIST repositories; full comparison in [MM_Wave1A_AVMNIST](https://github.com/Statera-Workshop/MM_Wave1A_AVMNIST)).
AV-MNIST (MultiBench pre-processed release): 28 x 28 PCA-reduced MNIST digit + 112 x 112 spectrogram of a spoken digit
with ESC-50 background noise, 10 classes; train 55,000 / val 5,000 (last 5,000 of the training file) / test 10,000 pairs.
Encoders from scratch: image LeNet(1,6,3) -> 48-d, audio LeNet(1,6,5) -> 192-d (MultiBench); image standardised with
the training mean / std, audio / 255. Adam 1e-3 constant, batch 64, budgets 30 and 100 epochs, seeds 777 / 778 / 779,
fp32, deterministic cuDNN. Every epoch: test accuracy and a frozen linear probe per modality (10,000 training pairs,
standardised features, full-batch Adam, test accuracy). Reported value = last epoch.

**Convergence measures.** Peak test accuracy and its epoch, drop after the peak (peak - final), and the *audio gap* =
audio-only model's probe - this model's audio probe (same budget; positive = audio under-learned in the fusion model).
*BN re-estimated*: the saved last-epoch weights evaluated again after recomputing BatchNorm statistics on 200 training
batches (the series' evaluation practice); for AV-MNIST it changes nothing beyond +/-0.004.

## Results (AV-MNIST, mean of seeds; range in brackets)

| Model | Optimiser | Epochs | Seeds | Final test acc (range) | Peak (epoch) | Drop after peak | Audio probe | Audio gap | Final, BN re-estimated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OGM-GE | Adam 1e-3 | 30 | 3 | 0.712 (0.001) | 0.721 (14.0) | 0.009 | 0.353 | +0.034 | 0.712 (0.006) |
| OGM-GE | Adam 1e-3 | 100 | 3 | 0.709 (0.004) | 0.721 (14.0) | 0.012 | 0.366 | +0.020 | 0.710 (0.003) |
| OGM-GE | SGD (official) | 30 | 3 | 0.698 (0.006) | 0.707 (17.7) | 0.009 | 0.307 | +0.080 | 0.699 (0.009) |
| OGM-GE | SGD (official) | 100 | 3 | 0.701 (0.006) | 0.708 (34.0) | 0.007 | 0.304 | +0.081 | 0.701 (0.007) |
| OGM (no GE noise) | Adam 1e-3 | 30 | 3 | 0.711 (0.011) | 0.722 (11.7) | 0.011 | 0.376 | +0.010 | 0.712 (0.013) |
| OGM (no GE noise) | Adam 1e-3 | 100 | 3 | 0.708 (0.005) | 0.722 (22.3) | 0.014 | 0.380 | +0.005 | 0.710 (0.008) |

Audio-only reference probe: 0.387 (30 epochs) / 0.385 (100 epochs).

**Finding.** On AV-MNIST the image is dominant throughout (score ratio about 2.5-2.9), so OGM scales the image gradients by about 0.75 and leaves audio at 1.0 - yet OGM-GE learns the audio *worse* than its unmodulated control (linear-head Late Fusion). Without the GE noise (OGM rows) the audio gap returns to the control's level or below: the under-learning comes from the noise added to the weak modality's encoder, not from the modulation.

## Weights

Release **v1.0** of this repository (18 files, SHA-256 in `SHA256SUMS.txt`): `avmnist_ogm_adam_e100_s777.pth`, `avmnist_ogm_adam_e100_s778.pth`, `avmnist_ogm_adam_e100_s779.pth`, `avmnist_ogm_adam_e30_s777.pth`, `avmnist_ogm_adam_e30_s778.pth`, `avmnist_ogm_adam_e30_s779.pth`, `avmnist_ogmge_adam_e100_s777.pth`, `avmnist_ogmge_adam_e100_s778.pth`, `avmnist_ogmge_adam_e100_s779.pth`, `avmnist_ogmge_adam_e30_s777.pth`, `avmnist_ogmge_adam_e30_s778.pth`, `avmnist_ogmge_adam_e30_s779.pth`, `avmnist_ogmge_sgd_official_e100_s777.pth`, `avmnist_ogmge_sgd_official_e100_s778.pth`, `avmnist_ogmge_sgd_official_e100_s779.pth`, `avmnist_ogmge_sgd_official_e30_s777.pth`, `avmnist_ogmge_sgd_official_e30_s778.pth`, `avmnist_ogmge_sgd_official_e30_s779.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_a as M
m = M.AVNet('<model>')  # img, aud, late, late_lin, tfn, lmf, gblend, ogmge, ogm
m.load_state_dict(torch.load('avmnist_<model>_<opt>_e<epochs>_s<seed>.pth', weights_only=True), strict=True)
```

**Licence note.** The AV-MNIST audio contains ESC-50 noise (CC BY-NC 3.0), so the released weights are for non-commercial use. See THIRD_PARTY.md.

## Run

Data: `gdown 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp -O avmnist.tar.gz` (MultiBench release), unpack to `~/MM_Wave1/data/avmnist`, check with `python code/count_avmnist.py ~/MM_Wave1/data/avmnist`.

```bash
cd code && bash smoke_a.sh   # gates + 1-epoch smoke
python mm_a.py --model ogmge --epochs 100 --seed 777
python analyze_a.py; python recal_bn_a.py
```
