# GAN - 生成對抗網絡

> **論文**: Generative Adversarial Networks
>
> **作者**: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
>
> **發表**: NIPS 2014
>
> **論文鏈接**: [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
>
> **引用次數**: 70,000+ (截至 2024)

---

## 🎯 簡介

**GAN (Generative Adversarial Network)** 是深度學習歷史上最具創造性的想法之一，由 Ian Goodfellow 於 2014 年提出。GAN 通過**兩個神經網絡的對抗訓練**，實現了高品質的資料生成，開創了生成模型的新時代。

Yann LeCun 曾評價 GAN 為："**近十年機器學習領域最有趣的想法**"。

### 核心思想

```
生成器 (Generator) vs 判別器 (Discriminator)

就像 "造假者" 和 "鑑定師" 的博弈：

造假者 (G):
- 目標: 生成逼真的假資料
- 希望: 騙過鑑定師

鑑定師 (D):
- 目標: 區分真假資料
- 希望: 識別所有假資料

通過不斷對抗，造假者越來越專業！
```

---

## 💡 核心創新

### 1. 對抗訓練框架

**兩個網絡的零和博弈**:

```python
# GAN 訓練偽程式碼
for epoch in epochs:
    for real_data in dataloader:
        # 1. 訓練判別器 D
        # 目標: max log(D(x)) + log(1 - D(G(z)))

        # 真實資料
        real_output = D(real_data)
        d_loss_real = -log(real_output)  # 希望 D(x) = 1

        # 生成假資料
        noise = random_noise()
        fake_data = G(noise)
        fake_output = D(fake_data.detach())
        d_loss_fake = -log(1 - fake_output)  # 希望 D(G(z)) = 0

        d_loss = d_loss_real + d_loss_fake
        update(D, d_loss)

        # 2. 訓練生成器 G
        # 目標: max log(D(G(z))) 或等價於 min log(1 - D(G(z)))

        fake_data = G(noise)
        fake_output = D(fake_data)
        g_loss = -log(fake_output)  # 希望 D(G(z)) = 1

        update(G, g_loss)
```

### 2. 數學原理

**目標函式**:
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

其中:
- D(x): 判別器對真實資料的輸出（希望接近 1）
- D(G(z)): 判別器對生成資料的輸出（D 希望接近 0，G 希望接近 1）
- z: 隨機噪聲
```

**理論保證**:
當 D 和 G 都達到最優時：
- D(x) = 1/2 對所有 x
- 生成分佈 p_g = 真實分佈 p_data

### 3. 訓練技巧

**1. 使用不同的生成器損失**:
```python
# 原始: min log(1 - D(G(z)))
# 問題: 梯度消失

# 改進: max log(D(G(z)))
g_loss = -torch.log(D(G(z)))
```

**2. 標籤平滑**:
```python
# 真實標籤不用 1，用 0.9
real_labels = 0.9

# 假標籤隨機在 0-0.1
fake_labels = torch.rand(batch_size) * 0.1
```

**3. 批量歸一化**:
```python
# 穩定訓練
self.bn = nn.BatchNorm2d(channels)
```

---

## 🏗️ 完整實現

### 基礎 GAN (MNIST)

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 訓練
generator = Generator()
discriminator = Discriminator()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # 真實和假標籤
        valid = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)

        # ---------------------
        #  訓練生成器
        # ---------------------
        optimizer_G.zero_grad()

        # 生成假圖像
        z = torch.randn(imgs.size(0), latent_dim)
        gen_imgs = generator(z)

        # 生成器損失
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  訓練判別器
        # ---------------------
        optimizer_D.zero_grad()

        # 真實圖像損失
        real_loss = adversarial_loss(discriminator(imgs), valid)
        # 假圖像損失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        # 總損失
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
```

### DCGAN (深度卷積 GAN)

```python
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()

        self.init_size = 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
```

---

## 📊 GAN 的演進

### GAN 家族樹

| 年份 | 模型 | 主要創新 |
|------|------|---------|
| 2014 | **GAN** | 對抗訓練框架 |
| 2015 | **DCGAN** | 深度卷積架構 |
| 2016 | **Pix2Pix** | 條件 GAN，圖像到圖像轉換 |
| 2017 | **CycleGAN** | 無配對資料的圖像轉換 |
| 2017 | **WGAN** | Wasserstein 距離，穩定訓練 |
| 2017 | **ProGAN** | 漸進式增長，高分辨率生成 |
| 2018 | **BigGAN** | 大規模訓練，ImageNet 生成 |
| 2018 | **StyleGAN** | 樣式遷移，高品質人臉生成 |
| 2019 | **StyleGAN2** | 改進的生成品質 |
| 2020 | **StyleGAN3** | 更好的平移等變性 |

---

## 🎯 應用場景

### 1. 圖像生成

```python
# 生成人臉
from stylegan2_pytorch import ModelLoader

loader = ModelLoader(
    base_dir = 'path/to/models',
    name = 'default'
)

noise = torch.randn(1, 512).cuda()
styles = loader.noise_to_styles(noise)
images = loader.styles_to_images(styles)
```

**應用**:
- 人臉生成（StyleGAN）
- 藝術創作
- 遊戲素材生成

### 2. 圖像轉換

```python
# Pix2Pix: 草圖 → 照片
input_sketch = load_sketch('sketch.png')
photo = pix2pix_model(input_sketch)

# CycleGAN: 馬 → 斑馬
horse_image = load_image('horse.jpg')
zebra_image = cyclegan_model(horse_image)
```

**應用**:
- 風格遷移
- 圖像修復
- 超分辨率（SRGAN）
- 黑白照片上色

### 3. 資料增強

```python
# 生成訓練資料
for i in range(1000):
    z = torch.randn(1, latent_dim)
    synthetic_data = generator(z)
    train_dataset.add(synthetic_data)
```

### 4. 異常檢測

```python
# 使用 GAN 進行異常檢測
reconstruction = G(D.encode(image))
anomaly_score = ||image - reconstruction||
```

---

## 🌟 GAN 的挑戰

### 1. 訓練不穩定

**問題**:
- 模式崩潰 (Mode Collapse)
- 梯度消失
- 難以收斂

**解決方案**:
- WGAN: Wasserstein 距離
- Spectral Normalization
- 自適應學習率

### 2. 評估困難

**問題**: 如何量化生成品質？

**指標**:
- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**
- **Precision & Recall**

```python
from pytorch_fid import fid_score

fid = fid_score.calculate_fid_given_paths(
    [real_images_path, generated_images_path],
    batch_size=50,
    device='cuda',
    dims=2048
)
```

### 3. 模式崩潰

**現象**: G 只生成少數幾種樣本

**解決方案**:
- Unrolled GAN
- Minibatch Discrimination
- 多個生成器

---

## 📚 參考資源

### 論文系列

1. **GAN** (2014): [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. **DCGAN** (2015): [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
3. **WGAN** (2017): [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
4. **StyleGAN** (2019): [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)

### 程式碼實現

- 🔥 **PyTorch GAN 集合**: [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
- 🎨 **StyleGAN2**: [NVlabs/stylegan2](https://github.com/NVlabs/stylegan2)
- 📦 **TensorFlow GAN**: [tensorflow/gan](https://github.com/tensorflow/gan)

### 學習資源

- 📖 **GAN 論文列表**: [hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
- 🎥 **Ian Goodfellow 演講**: [NIPS 2016 Tutorial](https://www.youtube.com/watch?v=AJVyzd0rqdc)
- 📚 **GAN 實戰**: [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

## 📝 引用

```bibtex
@article{goodfellow2014generative,
  title={Generative adversarial nets},
  author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  journal={Advances in neural information processing systems},
  volume={27},
  year={2014}
}
```

---

<div align="center">
  <p><strong>⭐ GAN: 開創生成模型新時代！</strong></p>
  <p>🎨 圖像生成 | 🔄 圖像轉換 | 🌟 無限可能</p>
  <p><i>最後更新: 2024-11-18</i></p>
</div>
