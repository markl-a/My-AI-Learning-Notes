# 在 Windows 11 上安裝 ROCm The Rock 完整指南

## 前言

AMD ROCm (Radeon Open Compute) 平台長期以來主要支援 Linux 系統，但隨著 The Rock 專案的推出，Windows 用戶終於有了一個輕量級的解決方案來開發 AMD GPU 加速應用程式。本文將詳細記錄在 Windows 11 上從零開始安裝 The Rock 的完整過程。

**發布日期**: 2025-10-25
**測試環境**: Windows 11 + AMD Radeon 8060S Graphics
**專案連結**: https://github.com/ROCm/TheRock

## 實際安裝記錄

本文是基於實際安裝過程的記錄，包含了遇到的問題和解決方案。

## 什麼是 The Rock？

The Rock 是 ROCm 團隊開發的輕量級開源建置平台，專門用於 HIP 和 ROCm 開發。主要特點包括：

- 支援 Windows 11 和多個 Linux 發行版
- 提供 ROCm 和 PyTorch 的 nightly builds
- 模組化架構，可選擇性安裝所需組件
- 完整的 CI/CD 自動化支援

## 系統需求

### 硬體需求
- **GPU**: AMD GPU (支援 ROCm 的型號)
  - 需要指定 GPU 架構，例如：gfx1100, gfx1101, gfx1102 等
- **記憶體**: 建議 16GB 以上
- **硬碟空間**: 至少 50GB 可用空間

### 軟體需求
- **作業系統**: Windows 11
- **開發工具**: Visual Studio 2022 (Community 版本即可)
- **Python**: 3.8 或更新版本
- **Git**: 用於下載源碼

## 安裝前準備

### 步驟 0: 系統編碼設定（重要！）

Windows 系統預設可能不是 UTF-8 編碼，這會導致編譯時出現問題。請按照以下步驟設定：

#### 方法一：永久設定系統編碼為 UTF-8（推薦）

1. **開啟控制台**
   - 按 `Win + R`，輸入 `control` 並按 Enter

2. **進入地區設定**
   - 點擊「時鐘和區域」→「地區」
   - 或直接搜尋「地區」

3. **變更系統地區設定**
   - 點擊「系統管理」標籤
   - 點擊「變更系統地區設定」按鈕
   - 勾選「使用 Unicode UTF-8 提供全球語言支援」
   - 點擊確定，系統會要求重新啟動

4. **重新啟動電腦**使設定生效

#### 方法二：臨時設定（每次開啟命令提示字元都要執行）

在 PowerShell 或命令提示字元中執行：
```powershell
# 設定當前視窗為 UTF-8 編碼
chcp 65001

# 驗證設定
chcp
# 應顯示: Active code page: 65001
```

#### 方法三：為 PowerShell 設定永久編碼

1. **建立或編輯 PowerShell 設定檔**
```powershell
# 檢查設定檔是否存在
Test-Path $PROFILE

# 如果不存在，創建它
if (!(Test-Path $PROFILE)) {
    New-Item -Type File -Path $PROFILE -Force
}

# 編輯設定檔
notepad $PROFILE
```

2. **在設定檔中加入以下內容**
```powershell
# 設定 UTF-8 編碼
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
```

3. **儲存並重新開啟 PowerShell**

#### 驗證編碼設定

執行以下命令確認設定成功：
```powershell
# 檢查當前編碼
[System.Text.Encoding]::Default

# 應該顯示 UTF-8 相關資訊
```

**注意**: 如果您的系統是非英文版 Windows，此步驟特別重要！

### 步驟 1: 安裝 Visual Studio 2022

1. 下載 [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
2. 安裝時選擇以下工作負載：
   - Desktop development with C++
   - Windows 11 SDK

### 步驟 2: 安裝必要工具

打開 PowerShell (管理員模式) 並執行：

```powershell
# 安裝 Chocolatey (如果還沒安裝)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 安裝必要工具
choco install git python cmake ninja -y

# 驗證安裝
git --version
python --version
cmake --version
ninja --version
```

### 步驟 3: 設定系統編碼 (重要!)

如果您的系統不是英文版，需要切換到 UTF-8 編碼：

```powershell
# 切換到 UTF-8 編碼
chcp 65001
```

## 下載並設定 The Rock

### 步驟 4: Clone Repository

```powershell
# 創建工作目錄
cd C:\
mkdir ROCm
cd ROCm

# Clone The Rock repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock
```

### 步驟 5: 設定 Python 虛擬環境

```powershell
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境 (Windows)
.venv\Scripts\Activate.bat

# 升級 pip
python -m pip install --upgrade pip

# 安裝 Python 依賴
pip install -r requirements.txt  # 如果存在 requirements.txt
```

## 建置 The Rock

### 步驟 6: 下載源碼

```powershell
# 使用 fetch_sources 腳本下載所需的源碼
python ./build_tools/fetch_sources.py
```

### 步驟 7: 配置 CMake

```powershell
# 創建 build 目錄
mkdir build
cd build

# 配置 CMake (請根據您的 GPU 型號調整)
cmake .. -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DTHEROCK_AMDGPU_TARGETS="gfx1100;gfx1101;gfx1102" ^
    -DCMAKE_INSTALL_PREFIX="C:/ROCm/TheRock/install"
```

**重要參數說明**:
- `-DTHEROCK_AMDGPU_TARGETS`: 指定您的 GPU 架構
  - gfx1100/1101/1102: RDNA 3 (RX 7900 系列)
  - gfx1030/1031/1032: RDNA 2 (RX 6000 系列)
  - gfx906/908: Vega 系列

### 步驟 8: 編譯

```powershell
# 開始編譯 (這可能需要很長時間)
ninja

# 或者使用 cmake 直接編譯
cmake --build . --config Release
```

### 步驟 9: 安裝

```powershell
# 安裝到指定目錄
ninja install

# 或
cmake --install . --config Release
```

## 驗證安裝

### 步驟 10: 測試 HIP 環境

```powershell
# 添加 The Rock 到系統 PATH
$env:PATH += ";C:\ROCm\TheRock\install\bin"

# 執行 hipinfo (如果有安裝)
hipinfo

# 檢查 GPU 架構
amdgpu-arch
```

### 步驟 11: 編寫測試程式

創建一個簡單的 HIP 程式 `test_hip.cpp`:

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);

    std::cout << "Found " << deviceCount << " HIP devices" << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        std::cout << "Device " << i << ": " << props.name << std::endl;
        std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "  Total Memory: " << props.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }

    return 0;
}
```

編譯並執行：

```powershell
# 編譯測試程式
hipcc test_hip.cpp -o test_hip.exe

# 執行
.\test_hip.exe
```

## 可選組件

The Rock 支援模組化安裝，您可以選擇性地啟用以下組件：

- **編譯器工具鏈**: LLVM/Clang for HIP
- **數學庫**:
  - BLAS (Basic Linear Algebra Subprograms)
  - SPARSE (稀疏矩陣運算)
- **深度學習**:
  - MIOpen (AMD 的深度學習原語庫)
  - PyTorch with ROCm backend
- **通訊庫**: RCCL (ROCm Communication Collectives Library)
- **效能分析工具**: ROCProfiler, ROCTracer

在 CMake 配置時添加對應的選項：

```powershell
cmake .. -G "Ninja" ^
    -DTHEROCK_ENABLE_MIOPEN=ON ^
    -DTHEROCK_ENABLE_PYTORCH=ON ^
    -DTHEROCK_ENABLE_RCCL=ON
```

## 常見問題與解決方案

### 問題 1: CMake 找不到編譯器

**解決方案**: 確保 Visual Studio 2022 已正確安裝，並在 VS 2022 Developer Command Prompt 中執行命令。

### 問題 2: GPU 架構不支援

**錯誤訊息**: `Unsupported GPU architecture`

**解決方案**: 檢查您的 GPU 型號並使用正確的 gfx 代碼：

```powershell
# 查詢 GPU 資訊
wmic path win32_VideoController get name
```

### 問題 3: 編譯記憶體不足

**解決方案**:
- 減少並行編譯任務數：`ninja -j4`
- 增加虛擬記憶體
- 關閉其他應用程式

### 問題 4: Python 腳本執行錯誤

**解決方案**:
- 確保虛擬環境已啟動
- 檢查 Python 版本是否符合要求
- 更新 pip 和所有依賴：`pip install --upgrade pip`

## 效能優化建議

1. **使用 DVC 加速 MIOpen**:
   ```powershell
   pip install dvc
   ```

2. **設定環境變數**:
   ```powershell
   # 設定 ROCm 路徑
   $env:ROCM_PATH = "C:\ROCm\TheRock\install"

   # 設定 HIP 路徑
   $env:HIP_PATH = "$env:ROCM_PATH\hip"
   ```

3. **使用 Release 模式編譯**: 始終使用 `-DCMAKE_BUILD_TYPE=Release` 以獲得最佳效能

## 後續開發

安裝完成後，您可以：

1. **開發 HIP 應用程式**: 使用 HIP API 開發跨平台 GPU 程式
2. **移植 CUDA 程式**: 使用 hipify 工具將 CUDA 程式碼轉換為 HIP
3. **訓練深度學習模型**: 使用 PyTorch with ROCm backend
4. **效能分析**: 使用 rocprof 和 roctracer 分析 GPU 程式效能

## 相關資源

- [ROCm 官方文檔](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [The Rock GitHub Issues](https://github.com/ROCm/TheRock/issues)
- [ROCm Community Forum](https://community.amd.com/t5/rocm/ct-p/amd-rocm)

## 實際安裝過程記錄

### 環境確認
```powershell
# GPU 檢查結果
AMD Radeon(TM) 8060S Graphics (Driver: 32.0.21025.10016)
# Strix Halo 平台, gfx1151 架構
# ROCm 6.4.1+ 支援

# 已安裝的工具
Git: version 2.50.1.windows.1
Python: 3.13.5
```

**重要發現**: AMD Radeon 8060S 是最新的 Strix Halo 平台 GPU (gfx1151)，在 ROCm 6.4.1 之後開始支援。The Rock 的 nightly builds 應該可以支援這個 GPU。

### Step 1: Clone Repository 
```powershell
git clone https://github.com/ROCm/TheRock.git
# 成功 clone，repository 大小約 50MB
```

### Step 2: 設定 Python 虛擬環境 
```powershell
cd TheRock
python -m venv .venv
# 使用 PowerShell 啟動虛擬環境
.\.venv\Scripts\Activate.ps1
# 安裝依賴
pip install -r requirements.txt
# 成功安裝所有 Python 套件，包含 meson, PyYAML, pytest-cmake 等
```

### Step 3: 下載子模組 
```powershell
# 設定 Git 長路徑支援
git config --global core.symlinks true
git config --global core.longpaths true

# 執行 fetch_sources.py
python ./build_tools/fetch_sources.py
```

**實際執行狀況**:
- 開始時間: 12:50 PM
- 完成時間: 1:09 PM (約 20 分鐘)
- 已下載的子模組：
  ```
  ✓ base/amdsmi
  ✓ base/half
  ✓ base/rocm-cmake
  ✓ comm-libs/rccl
  ✓ comm-libs/rccl-tests
  ✓ compiler/amd-llvm (LLVM 專案，最大)
  ✓ compiler/hipify
  ✓ compiler/spirv-llvm-translator
  ✓ rocm-libraries
  ✓ rocm-systems
  ```

**注意事項**:
- fetch_sources.py 會下載多個大型子模組，包括 LLVM 專案
- 下載時間取決於網路速度，可能需要 30 分鐘到數小時
- LLVM 專案特別大（可能超過 1GB）
- 建議在網路穩定的環境下進行

### Step 4: 安裝必要的工具

已安裝的工具：
1. **Visual Studio 2022 Community** ✓
   - 版本: MSVC 14.44.35207
   - 路徑: C:\Program Files\Microsoft Visual Studio\2022\Community

2. **CMake** ✓
   - 版本: 4.1.2
   - 安裝命令: `winget install Kitware.CMake`
   - 安裝時間: 約 1 分鐘

3. **Ninja** ✓
   - 版本: 1.13.1
   - 安裝命令: `winget install Ninja-build.Ninja`
   - 安裝時間: 約 30 秒

### Step 5: 配置 CMake 

```powershell
# 使用 Visual Studio 2022 Generator (x64)
cmake -B build -G "Visual Studio 17 2022" -A x64 . -DTHEROCK_AMDGPU_TARGETS="gfx1151" -DCMAKE_BUILD_TYPE=Release
```

**配置成功！重要資訊：**
- ✓ 偵測到 AMD Strix Halo iGPU (gfx1151)
- ✓ ROCm version: 7.10.0
- ✓ HIP version: 7.1.0
- ✓ 啟用的功能：
  - COMPILER (LLVM)
  - HIP_RUNTIME
  - HIPIFY
  - BLAS, FFT, SPARSE, SOLVER
  - MIOpen (深度學習)
- 配置時間：約 6 秒
- Build files 位置：C:/Users/m4932/Documents/test/TheRock/build

### 遇到的問題與解決

1. **DVC 和 dvc-s3 安裝**
   - fetch_sources.py 需要 DVC 來下載大型文件
   - 解決方案：
   ```powershell
   pip install dvc dvc-s3
   ```

2. **CMake 配置問題 - 32-bit vs 64-bit**
   - 錯誤：`Cannot build 32-bit ROCm with MSVC`
   - 原因：CMake 偵測到 32-bit 編譯器
   - 解決：使用 Visual Studio Generator 並指定 x64 架構
   ```powershell
   cmake -G "Visual Studio 17 2022" -A x64
   ```

3. **Ninja 找不到**
   - 錯誤：`CMAKE_MAKE_PROGRAM is not set`
   - 解決：改用 Visual Studio Generator 而非 Ninja

4. **Windows 特有注意事項**
   - 必須啟用長路徑支援
   - 需要使用 UTF-8 編碼 (chcp 65001)
   - 建議使用 Dev Drive 提升效能

## 下一步：編譯 The Rock

配置成功後，可以開始編譯：

```powershell
# 使用 Visual Studio 編譯 (會需要很長時間)
cmake --build build --config Release

# 或者開啟 Visual Studio 直接編譯
# 開啟 TheRock.sln 檔案
```

**重要提醒**：
- 完整編譯可能需要 2-6 小時（取決於硬體）
- 需要大約 100GB+ 的硬碟空間
- 建議有 32GB+ RAM 以獲得最佳編譯速度

## 支援的 Python 套件和框架

成功安裝 The Rock 後，AMD Radeon 8060S (gfx1151) 將支援以下 Python 套件：

### 深度學習框架

#### 1. PyTorch with ROCm
```python
# 安裝
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 測試
import torch
print(torch.cuda.is_available())  # HIP/ROCm 相容 CUDA API
print(torch.cuda.get_device_name(0))  # 顯示 AMD Radeon 8060S
```

#### 2. TensorFlow with ROCm
```python
# 安裝
pip install tensorflow-rocm

# 測試
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

#### 3. JAX with ROCm
```python
# 安裝
pip install jax[rocm]

# 測試
import jax
print(jax.devices())
```

### 支援的應用場景

#### 生成式 AI 模型
- **Stable Diffusion** - 圖像生成
- **LLaMA** - 大型語言模型
- **Whisper** - 語音識別
- **BERT/GPT** - 自然語言處理

#### 電腦視覺
- **YOLO** - 物體偵測
- **Detectron2** - 實例分割
- **timm** - PyTorch Image Models
- **OpenCV GPU** - 影像處理

#### 科學計算
- **CuPy for ROCm** - GPU 加速的 NumPy
```python
pip install cupy-rocm-5-0
import cupy as cp
x_gpu = cp.array([1, 2, 3])
```

- **RAPIDS** (部分支援) - GPU 資料科學
- **XGBoost GPU** - 梯度提升

### GPU 功能測試腳本

```python
# test_amd_gpu.py
import sys
import platform

print("=== 系統資訊 ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"GPU: AMD Radeon 8060S (gfx1151)")

# PyTorch 測試
try:
    import torch
    print("\n=== PyTorch ROCm ===")
    print(f"版本: {torch.__version__}")
    print(f"GPU 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU 數量: {torch.cuda.device_count()}")
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")

        # GPU 記憶體資訊
        props = torch.cuda.get_device_properties(0)
        print(f"GPU 記憶體: {props.total_memory / 1024**3:.2f} GB")

        # 簡單效能測試
        x = torch.randn(5000, 5000).cuda()
        y = torch.randn(5000, 5000).cuda()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        print("矩陣運算測試: ✓ 成功")
except Exception as e:
    print(f"PyTorch 測試失敗: {e}")

# TensorFlow 測試
try:
    import tensorflow as tf
    print("\n=== TensorFlow ROCm ===")
    print(f"版本: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"偵測到 {len(gpus)} 個 GPU")
    for gpu in gpus:
        print(f"  - {gpu}")
except Exception as e:
    print(f"TensorFlow 測試失敗: {e}")
```

### 效能預期

根據 AMD Radeon 8060S (Strix Halo) 規格：
- **2560 個著色單元**
- **40 個光追核心**
- **共享系統記憶體** (最高可用 32GB+)

預期效能：
- **Stable Diffusion**: 512x512 圖像生成約 5-10 秒
- **LLaMA 7B**: 推理速度約 20-30 tokens/秒
- **PyTorch 訓練**: 比 CPU 快 10-50 倍
- **矩陣運算**: FP32 效能約 10-15 TFLOPS

### 注意事項

1. **版本相容性**
   - 確保 Python 套件版本與 ROCm 7.10.0 相容
   - 某些 CUDA 專屬功能可能需要調整

2. **記憶體管理**
   - iGPU 共享系統記憶體
   - 建議保留至少 8GB 給系統使用

3. **最佳化建議**
   - 使用 mixed precision (FP16) 提升效能
   - 批次大小 (batch size) 根據記憶體調整
   - 啟用 Flash Attention 等優化技術

## 總結

### 成功完成的步驟
1. ✅ 環境準備 - Visual Studio 2022, CMake, Ninja
2. ✅ Clone The Rock repository
3. ✅ 下載所有子模組（包括 LLVM）
4. ✅ 安裝 Python 依賴（DVC, dvc-s3）
5. ✅ CMake 配置成功（偵測到 gfx1151）

### 關鍵發現
- **AMD Radeon 8060S (Strix Halo)** 是最新的 GPU，使用 gfx1151 架構
- ROCm 7.10.0 支援這個 GPU
- Windows 編譯需要使用 Visual Studio 2022 x64 工具鏈
- The Rock 提供了完整的 ROCm 開發環境，包括 HIP、MIOpen 等

### 時間統計
- Repository clone: 1 分鐘
- 子模組下載: 20 分鐘
- 工具安裝: 5 分鐘
- CMake 配置: 6 秒
- 預計編譯時間: 2-6 小時

## 結語

The Rock 為 Windows 用戶提供了進入 ROCm 生態系統的重要途徑。雖然安裝過程需要一些技術知識，但完成後您將擁有一個完整的 AMD GPU 開發環境，可以開發 HIP 應用程式、使用 MIOpen 進行深度學習等。

對於 AMD Radeon 8060S (Strix Halo) 用戶來說，這是目前在 Windows 上使用 ROCm 的最佳選擇。

---



**相關資源**:
- [The Rock GitHub](https://github.com/ROCm/TheRock)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

**聲明**: 本文記錄了實際安裝過程，步驟可能因版本更新而有所變化。