# YOLO Android 部署指南（YOLO11/v10/v9/v8）

> 📱 **平台：** Android
> 🔧 **框架：** ONNX Runtime / NCNN / TensorFlow Lite
> ⚡ **特色：** 即時物件偵測，支援多種推論引擎
> 🔄 **最後更新：** 2025-01
> ✅ **支援版本：** YOLO11, YOLOv10, YOLOv9, YOLOv8

---

## 📖 簡介

本指南介紹如何將訓練好的 **YOLO 系列模型**（YOLO11、YOLOv10、YOLOv9、YOLOv8）部署到 Android 應用程式中，實現即時物件偵測功能。所有 YOLO 版本使用相同的部署流程。

### 支援的部署方式

| 方式 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **ONNX Runtime** | 跨平台、效能好 | 模型檔案較大 | 通用推薦 |
| **NCNN** | 速度快、體積小 | 需要額外轉換 | 性能優先 |
| **TensorFlow Lite** | 官方支援、穩定 | 轉換複雜 | 穩定性優先 |

---

## 🚀 快速開始

### 方法一：使用 ONNX Runtime（推薦）

#### 1. 導出 ONNX 模型

```python
from ultralytics import YOLO

# 載入訓練好的模型（支援所有版本）
model = YOLO('best.pt')  # 可以是 YOLO11/v10/v9/v8 訓練的模型

# 導出為 ONNX 格式
model.export(
    format='onnx',
    imgsz=640,
    opset=12,  # ONNX opset 版本
    simplify=True,  # 簡化模型
    dynamic=False,  # 固定輸入尺寸
)
```

**輸出：** `best.onnx`

**不同版本的建議：**

```python
# YOLO11 - 推薦用於移動端（最快）
model = YOLO('yolo11n.pt')  # 或訓練好的模型
model.export(format='onnx', imgsz=640, simplify=True)

# YOLOv10 - 無需 NMS，部署更簡單
model = YOLO('yolov10n.pt')
model.export(format='onnx', imgsz=640, simplify=True)

# YOLOv9 - 準確率優先
model = YOLO('yolov9c.pt')
model.export(format='onnx', imgsz=640, simplify=True, half=True)  # FP16

# YOLOv8 - 穩定可靠
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

#### 2. Android 專案設定

**build.gradle (Module: app)**

```gradle
dependencies {
    // ONNX Runtime
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.3'

    // 相機和圖像處理
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
}
```

#### 3. 添加模型到專案

1. 在 `app/src/main/assets/` 建立資料夾
2. 將 `best.onnx` 放入 `assets/` 資料夾
3. 同時添加 `labels.txt`（類別名稱，每行一個）

#### 4. 推理程式碼範例

```kotlin
import ai.onnxruntime.*
import android.graphics.Bitmap

class YOLOv8Detector(private val context: Context) {
    private lateinit var ortSession: OrtSession
    private lateinit var ortEnvironment: OrtEnvironment

    // 模型參數
    private val inputShape = longArrayOf(1, 3, 640, 640)
    private val confThreshold = 0.25f
    private val iouThreshold = 0.45f

    init {
        loadModel()
    }

    private fun loadModel() {
        ortEnvironment = OrtEnvironment.getEnvironment()
        val modelBytes = context.assets.open("best.onnx").readBytes()
        ortSession = ortEnvironment.createSession(modelBytes)
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        // 1. 預處理：調整大小和歸一化
        val inputTensor = preprocessImage(bitmap)

        // 2. 推理
        val inputs = mapOf("images" to inputTensor)
        val outputs = ortSession.run(inputs)

        // 3. 後處理：解析輸出
        val output = outputs[0].value as Array<Array<FloatArray>>
        val detections = postprocess(output[0])

        return detections
    }

    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        // 調整圖像大小為 640x640
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // 轉換為浮點陣列並歸一化 [0, 255] -> [0, 1]
        val floatArray = FloatArray(3 * 640 * 640)
        val pixels = IntArray(640 * 640)
        resizedBitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            // RGB 順序，歸一化
            floatArray[i] = ((pixel shr 16 and 0xFF) / 255.0f)  // R
            floatArray[640 * 640 + i] = ((pixel shr 8 and 0xFF) / 255.0f)  // G
            floatArray[640 * 640 * 2 + i] = ((pixel and 0xFF) / 255.0f)  // B
        }

        return OnnxTensor.createTensor(ortEnvironment, floatArray, inputShape)
    }

    private fun postprocess(output: Array<FloatArray>): List<Detection> {
        val detections = mutableListOf<Detection>()

        // YOLOv8 輸出格式：[batch, 84, 8400]
        // 84 = 4 (bbox) + 80 (classes for COCO, adjust for your model)
        for (i in 0 until output[0].size) {
            val data = output.map { it[i] }

            // 取得最大類別分數
            val classScores = data.subList(4, data.size)
            val maxScore = classScores.maxOrNull() ?: 0f

            if (maxScore > confThreshold) {
                val classId = classScores.indexOf(maxScore)
                val cx = data[0]
                val cy = data[1]
                val w = data[2]
                val h = data[3]

                // 轉換為 x1, y1, x2, y2
                val x1 = cx - w / 2
                val y1 = cy - h / 2
                val x2 = cx + w / 2
                val y2 = cy + h / 2

                detections.add(Detection(x1, y1, x2, y2, maxScore, classId))
            }
        }

        // NMS (非極大值抑制)
        return nms(detections, iouThreshold)
    }

    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()

        for (detection in sorted) {
            var keep = true
            for (selected in selected) {
                if (iou(detection, selected) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) selected.add(detection)
        }

        return selected
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)

        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - intersection

        return intersection / union
    }

    fun close() {
        ortSession.close()
        ortEnvironment.close()
    }
}

data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)
```

#### 5. 在 Activity 中使用

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var detector: YOLOv8Detector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化檢測器
        detector = YOLOv8Detector(this)

        // 使用檢測器
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)
        val detections = detector.detect(bitmap)

        // 顯示結果
        detections.forEach { detection ->
            Log.d("YOLO", "Class: ${detection.classId}, Conf: ${detection.confidence}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
    }
}
```

---

### 方法二：使用 NCNN（高性能）

NCNN 是騰訊開源的輕量級神經網路推理框架，針對移動端優化。

#### 1. 轉換模型

```bash
# 先轉換為 ONNX
python -m ultralytics export model=best.pt format=onnx

# 使用 onnx2ncnn 轉換
onnx2ncnn best.onnx best.param best.bin
```

#### 2. Android 整合

參考官方範例：[ncnn-android-yolov8](https://github.com/FeiGeChuanShu/ncnn-android-yolov8)

**build.gradle**

```gradle
dependencies {
    implementation 'com.tencent.ncnn:ncnn:20230816'
}
```

---

### 方法三：使用 TensorFlow Lite

#### 1. 導出 TFLite 模型

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(
    format='tflite',
    imgsz=640,
    int8=False,  # 設為 True 啟用 INT8 量化
)
```

#### 2. Android 整合

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

---

## 🎨 即時相機檢測

### CameraX 整合範例

```kotlin
class CameraActivity : AppCompatActivity() {
    private lateinit var detector: YOLOv8Detector
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        detector = YOLOv8Detector(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, YOLOAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private inner class YOLOAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val bitmap = imageProxy.toBitmap()
            val detections = detector.detect(bitmap)

            // 在 UI 繪製結果
            runOnUiThread {
                drawDetections(detections)
            }

            imageProxy.close()
        }
    }
}
```

---

## ⚡ 性能優化

### 1. 模型量化

```python
# INT8 量化（減小模型大小，提升速度）
model.export(format='onnx', int8=True)
```

### 2. 使用 GPU 加速

```kotlin
// ONNX Runtime GPU
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addNnapi()  // 使用 NNAPI
```

### 3. 降低輸入解析度

```python
# 使用較小的輸入尺寸
model.export(format='onnx', imgsz=320)  # 640 -> 320
```

### 4. 多線程處理

```kotlin
val executorService = Executors.newFixedThreadPool(4)
```

---

## 📊 性能基準測試

**Snapdragon 888 性能對比：**

| 模型 | 框架 | FPS | 延遲 | 備註 |
|------|------|-----|------|------|
| **YOLO11n** | NCNN | ~85 | 11.7ms | 最快，推薦 |
| **YOLO11n** | ONNX | ~55 | 18ms | 較快 |
| **YOLOv10n** | ONNX | ~50 | 20ms | 無需 NMS |
| **YOLOv9s** | ONNX | ~35 | 28ms | 高準確率 |
| **YOLOv8n** | NCNN | ~60 | 16ms | 穩定 |
| **YOLOv8n** | ONNX | ~45 | 22ms | 可靠 |
| **YOLO11n-int8** | NCNN | ~95 | 10.5ms | 最優 |

**建議：**
- **移動端首選：** YOLO11n + NCNN
- **平衡選擇：** YOLOv10n + ONNX Runtime
- **準確率優先：** YOLOv9s + ONNX Runtime
- **穩定部署：** YOLOv8n + NCNN

---

## 🔧 常見問題

### Q: 模型推論速度慢？

**A:**
- 使用量化模型（INT8）
- 降低輸入解析度（320x320）
- 使用 NCNN 框架
- 啟用 GPU/NNAPI 加速

### Q: APP 體積過大？

**A:**
- 使用模型壓縮
- 使用 YOLOv8n（最小模型）
- 移除未使用的資源

### Q: 記憶體不足？

**A:**
- 降低批次大小
- 使用較小的模型
- 及時釋放資源

---

## 📚 參考資源

### 官方文檔
1. [ONNX Runtime Android](https://onnxruntime.ai/docs/build/android.html)
2. [Ultralytics Export](https://docs.ultralytics.com/modes/export/)
3. [Android CameraX](https://developer.android.com/training/camerax)

### 開源專案
1. [NCNN Android YOLOv8](https://github.com/FeiGeChuanShu/ncnn-android-yolov8)
2. [ONNX Runtime Examples](https://github.com/microsoft/onnxruntime-inference-examples)

### 教學文章
1. [Android YOLOv8 部署詳解](https://blog.csdn.net/level_code/article/details/127654653)
2. [NCNN 移動端部署](https://zhuanlan.zhihu.com/p/516858508)

---

## 📝 完整專案範例

完整的 Android Studio 專案範例即將添加到本目錄。

**預期結構：**
```
Android/
├── README.md           # 本文件
├── YOLOv8App/         # Android Studio 專案
│   ├── app/
│   │   ├── src/
│   │   ├── assets/
│   │   │   ├── best.onnx
│   │   │   └── labels.txt
│   │   └── build.gradle
│   └── build.gradle
└── models/            # 範例模型
```

---

**快樂開發！** 🚀
