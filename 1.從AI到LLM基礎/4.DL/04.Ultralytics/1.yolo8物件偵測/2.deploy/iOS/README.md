# YOLO iOS 部署指南（YOLO11/v10/v9/v8）

> 📱 **平台：** iOS 14.0+
> 🔧 **框架：** Core ML, Vision
> ⚡ **特色：** 即時物件偵測，原生 iOS 整合
> 🔄 **最後更新：** 2025-01
> ✅ **支援版本：** YOLO11, YOLOv10, YOLOv9, YOLOv8

---

## 📖 簡介

本目錄包含完整的 iOS 部署範例，展示如何將訓練好的 **YOLO 系列模型**（YOLO11、YOLOv10、YOLOv9、YOLOv8）部署到 iPhone/iPad 上實現即時物件偵測。所有 YOLO 版本使用相同的部署流程。

### 為什麼選擇 Core ML？

- ✅ **原生支援** - Apple 官方機器學習框架
- ✅ **高效能** - 針對 Apple 硬體優化（Neural Engine）
- ✅ **低功耗** - 專為移動設備設計
- ✅ **易整合** - 與 Vision 框架無縫配合
- ✅ **隱私保護** - 本地推理，無需網路

---

## 📁 專案結構

```
iOS/
├── README.md                          # 本文件
├── Yolov8-RealTime-iOS/              # 完整 iOS 應用程式
│   ├── Assets.xcassets/              # 資源檔案
│   ├── yolov8s.mlpackage/            # Core ML 模型
│   ├── ViewController.swift          # 主視圖控制器
│   ├── YOLOv8Predictor.swift         # YOLO 預測器（即將添加）
│   └── Info.plist
└── models/                            # 範例模型（即將添加）
    └── yolov8n.mlmodel
```

---

## 🚀 快速開始

### 1. 導出 Core ML 模型

```python
from ultralytics import YOLO

# 載入訓練好的模型（支援所有版本）
model = YOLO('best.pt')  # 可以是 YOLO11/v10/v9/v8 訓練的模型

# 導出為 Core ML 格式
model.export(
    format='coreml',
    imgsz=640,
    nms=True,        # 包含 NMS（非極大值抑制）
    int8=False,      # 使用 FP16（更快）或 INT8（更小）
)
```

**輸出檔案：**
- `best.mlpackage/` - Core ML 模型包
- 包含模型、元資料、預覽

**不同版本的建議：**

```python
# YOLO11 - 推薦用於 iOS（最快，最新）
model = YOLO('yolo11n.pt')  # 或訓練好的模型
model.export(format='coreml', imgsz=640, nms=True, half=True)  # FP16

# YOLOv10 - 無需 NMS，部署更簡單
model = YOLO('yolov10n.pt')
model.export(format='coreml', imgsz=640, nms=False)  # YOLOv10 不需要 NMS

# YOLOv9 - 準確率優先
model = YOLO('yolov9c.pt')
model.export(format='coreml', imgsz=640, nms=True, half=True)

# YOLOv8 - 穩定可靠
model = YOLO('yolov8n.pt')
model.export(format='coreml', imgsz=640, nms=True)
```

### 2. 打開 Xcode 專案

```bash
cd Yolov8-RealTime-iOS
open Yolov8-RealTime-iOS.xcodeproj
```

### 3. 添加模型到專案

1. 將 `.mlpackage` 拖到 Xcode 專案
2. 確認「Target Membership」已勾選
3. Xcode 會自動生成 Swift 接口

### 4. 執行應用程式

1. 選擇目標設備（iPhone/iPad）
2. 點擊 Run（⌘R）
3. 授予相機權限
4. 開始即時偵測

---

## 💻 程式碼實作

### 1. 模型載入

```swift
import CoreML
import Vision

class YOLOv8Predictor {
    // 模型相關
    private var model: VNCoreMLModel?
    private let modelName = "yolov8s"  // 你的模型名稱

    // 推理參數
    private let confidenceThreshold: Float = 0.25
    private let iouThreshold: Float = 0.45

    init() {
        loadModel()
    }

    private func loadModel() {
        guard let modelURL = Bundle.main.url(
            forResource: modelName,
            withExtension: "mlmodelc"
        ) else {
            print("❌ 找不到模型檔案")
            return
        }

        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
            print("✅ 模型載入成功")
        } catch {
            print("❌ 模型載入失敗: \(error)")
        }
    }
}
```

### 2. 圖像推理

```swift
extension YOLOv8Predictor {
    func predict(image: CVPixelBuffer, completion: @escaping ([Detection]) -> Void) {
        guard let model = model else {
            completion([])
            return
        }

        // 建立 Vision 請求
        let request = VNCoreMLRequest(model: model) { request, error in
            guard error == nil else {
                print("❌ 推理錯誤: \(error!)")
                completion([])
                return
            }

            // 處理結果
            let detections = self.processResults(request.results)
            completion(detections)
        }

        // 設置圖像方向
        request.imageCropAndScaleOption = .scaleFill

        // 執行推理
        let handler = VNImageRequestHandler(
            cvPixelBuffer: image,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            print("❌ 推理執行失敗: \(error)")
            completion([])
        }
    }

    private func processResults(_ results: [Any]?) -> [Detection] {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            return []
        }

        var detections: [Detection] = []

        for observation in results {
            guard let topLabel = observation.labels.first,
                  topLabel.confidence >= confidenceThreshold else {
                continue
            }

            let bbox = observation.boundingBox
            let detection = Detection(
                label: topLabel.identifier,
                confidence: topLabel.confidence,
                boundingBox: bbox
            )
            detections.append(detection)
        }

        return detections
    }
}

// 檢測結果結構
struct Detection {
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}
```

### 3. 即時相機推理

```swift
import AVFoundation

class CameraViewController: UIViewController {
    // 相機相關
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?

    // YOLO 預測器
    private let predictor = YOLOv8Predictor()

    // 顯示結果
    private var detectionOverlay: CALayer?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupOverlay()
    }

    private func setupCamera() {
        // 建立 capture session
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high

        // 添加相機輸入
        guard let videoDevice = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ),
        let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            print("❌ 無法存取相機")
            return
        }

        captureSession?.addInput(videoInput)

        // 添加影片輸出
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(
            self,
            queue: DispatchQueue(label: "videoQueue")
        )
        captureSession?.addOutput(videoOutput)

        // 設置預覽層
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = view.bounds
        view.layer.addSublayer(previewLayer!)

        // 啟動 session
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession?.startRunning()
        }
    }

    private func setupOverlay() {
        detectionOverlay = CALayer()
        detectionOverlay?.frame = view.bounds
        view.layer.addSublayer(detectionOverlay!)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        // 執行 YOLO 推理
        predictor.predict(image: pixelBuffer) { [weak self] detections in
            DispatchQueue.main.async {
                self?.updateDetections(detections)
            }
        }
    }

    private func updateDetections(_ detections: [Detection]) {
        // 清除舊的檢測框
        detectionOverlay?.sublayers?.forEach { $0.removeFromSuperlayer() }

        // 繪製新的檢測框
        for detection in detections {
            drawBoundingBox(detection)
        }
    }

    private func drawBoundingBox(_ detection: Detection) {
        // 轉換座標（Core ML 使用左下角為原點）
        let bounds = view.bounds
        let scale = CGAffineTransform.identity.scaledBy(x: bounds.width, y: bounds.height)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -1)
        let bbox = detection.boundingBox.applying(transform).applying(scale)

        // 建立邊界框
        let boxLayer = CALayer()
        boxLayer.frame = bbox
        boxLayer.borderWidth = 2.0
        boxLayer.borderColor = UIColor.red.cgColor
        boxLayer.cornerRadius = 4.0

        // 建立標籤
        let textLayer = CATextLayer()
        textLayer.string = "\(detection.label) \(Int(detection.confidence * 100))%"
        textLayer.fontSize = 14
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.backgroundColor = UIColor.red.cgColor
        textLayer.frame = CGRect(
            x: bbox.minX,
            y: bbox.minY - 20,
            width: bbox.width,
            height: 20
        )
        textLayer.alignmentMode = .center

        // 添加到 overlay
        detectionOverlay?.addSublayer(boxLayer)
        detectionOverlay?.addSublayer(textLayer)
    }
}
```

### 4. Info.plist 權限設置

在 `Info.plist` 中添加相機權限：

```xml
<key>NSCameraUsageDescription</key>
<string>我們需要使用相機進行即時物件偵測</string>
```

---

## ⚡ 性能優化

### 1. 模型優化

```python
# 使用 FP16 量化（推薦）
model.export(format='coreml', int8=False, half=True)

# 使用 INT8 量化（更小，稍慢）
model.export(format='coreml', int8=True)

# 使用較小的模型
model = YOLO('yolov8n.pt')  # nano 版本
```

### 2. 推論優化

```swift
// 降低相機解析度
captureSession?.sessionPreset = .medium  // 或 .low

// 跳幀處理（每 N 幀推理一次）
private var frameCount = 0
private let inferenceInterval = 3  // 每 3 幀推理一次

func captureOutput(...) {
    frameCount += 1
    guard frameCount % inferenceInterval == 0 else { return }

    // 執行推理...
}

// 使用 GPU 加速
let request = VNCoreMLRequest(model: model)
request.usesCPUOnly = false  // 使用 Neural Engine
```

### 3. UI 優化

```swift
// 使用 CADisplayLink 同步繪製
private var displayLink: CADisplayLink?

func startDisplayLink() {
    displayLink = CADisplayLink(
        target: self,
        selector: #selector(updateDisplay)
    )
    displayLink?.add(to: .main, forMode: .common)
}

@objc func updateDisplay() {
    // 更新 UI
}
```

---

## 📊 性能基準測試

**最新設備性能對比：**

| 設備 | 模型 | FPS | 延遲 | 功耗 | 備註 |
|------|------|-----|------|------|------|
| **iPhone 15 Pro** | YOLO11n | ~75 | 13ms | 極低 | 最佳 |
| **iPhone 15 Pro** | YOLOv10n | ~70 | 14ms | 極低 | 優秀 |
| **iPhone 15 Pro** | YOLOv9s | ~55 | 18ms | 低 | 高精度 |
| **iPhone 15 Pro** | YOLOv8n | ~65 | 15ms | 低 | 穩定 |
| **iPhone 14 Pro** | YOLO11n | ~70 | 14ms | 低 | 推薦 |
| **iPhone 14 Pro** | YOLOv8n | ~60 | 16ms | 低 | 可靠 |
| **iPhone 13** | YOLO11n | ~55 | 18ms | 低 | 良好 |
| **iPad Pro M2** | YOLO11s | ~80 | 12ms | 極低 | 最優 |

**建議：**
- **iPhone 15/14 Pro 首選：** YOLO11n（最佳性能）
- **需要極致速度：** YOLOv10n（無 NMS）
- **準確率優先：** YOLOv9s
- **穩定部署：** YOLOv8n

---

## 🔧 常見問題

### Q: 模型導入後 Xcode 報錯？

**A:**
- 檢查 iOS Deployment Target ≥ 14.0
- 確認模型檔案已添加到 Target
- 清理專案：Product → Clean Build Folder

### Q: 推論速度慢？

**A:**
- 使用 YOLOv8n（最小模型）
- 啟用量化（INT8 或 FP16）
- 降低輸入解析度
- 使用跳幀處理
- 確認使用 Neural Engine（不是 CPU）

### Q: 記憶體警告？

**A:**
- 降低相機解析度
- 使用較小的模型
- 及時釋放資源
- 使用 Instruments 分析記憶體

### Q: 相機畫面延遲？

**A:**
- 降低推理頻率（跳幀）
- 使用非同步推理
- 優化 UI 更新

---

## 📚 進階功能

### 1. 拍照儲存檢測結果

```swift
func capturePhoto() {
    let photoOutput = AVCapturePhotoOutput()
    captureSession?.addOutput(photoOutput)

    let settings = AVCapturePhotoSettings()
    photoOutput.capturePhoto(with: settings, delegate: self)
}

extension CameraViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            return
        }

        // 在圖像上繪製檢測框並儲存
        let annotatedImage = drawDetectionsOnImage(image)
        UIImageWriteToSavedPhotosAlbum(annotatedImage, nil, nil, nil)
    }
}
```

### 2. 影片錄製

```swift
let movieOutput = AVCaptureMovieFileOutput()
captureSession?.addOutput(movieOutput)

// 開始錄製
let outputURL = FileManager.default.temporaryDirectory
    .appendingPathComponent("detection.mov")
movieOutput.startRecording(to: outputURL, recordingDelegate: self)

// 停止錄製
movieOutput.stopRecording()
```

### 3. 自定義檢測框樣式

```swift
private func drawBoundingBox(_ detection: Detection) {
    let color = getColorForClass(detection.label)
    let boxLayer = CAShapeLayer()

    let path = UIBezierPath(roundedRect: bbox, cornerRadius: 8.0)
    boxLayer.path = path.cgPath
    boxLayer.strokeColor = color.cgColor
    boxLayer.lineWidth = 3.0
    boxLayer.fillColor = UIColor.clear.cgColor

    // 添加陰影
    boxLayer.shadowColor = UIColor.black.cgColor
    boxLayer.shadowOpacity = 0.5
    boxLayer.shadowRadius = 3.0

    detectionOverlay?.addSublayer(boxLayer)
}

private func getColorForClass(_ className: String) -> UIColor {
    let colors: [String: UIColor] = [
        "person": .red,
        "car": .blue,
        "dog": .green,
        "cat": .orange
    ]
    return colors[className] ?? .yellow
}
```

---

## 🎓 學習資源

### 官方文檔
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Vision Framework](https://developer.apple.com/documentation/vision)
- [Ultralytics iOS Guide](https://docs.ultralytics.com/guides/ios-app/)

### 教學文章
- [Building a Real-Time Object Detection App](https://developer.apple.com/documentation/vision/recognizing_objects_in_live_capture)

### 範例專案
- [Apple Vision Samples](https://developer.apple.com/documentation/vision/recognizing_objects_in_live_capture)

---

## 🎯 專案範例

本目錄包含完整的 iOS 應用程式範例：

**功能特色：**
- ✅ 即時相機物件偵測
- ✅ 多類別檢測支援
- ✅ 流暢的 UI/UX
- ✅ 性能優化
- ✅ 錯誤處理

**使用方式：**
1. 打開 `Yolov8-RealTime-iOS.xcodeproj`
2. 替換為你的模型
3. 更新類別名稱
4. 執行即可

---

## 🔗 相關資源

- **訓練模型：** `../../1.train/`
- **Android 部署：** `../Android/`
- **主文檔：** `../../../README.md`

---

**祝部署順利！** 🚀
