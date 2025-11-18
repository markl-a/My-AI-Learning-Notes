#!/usr/bin/env python3
"""
YOLOv8 訓練腳本

使用方式:
    python train_yolov8.py --data dataset.yaml --model yolov8n.pt --epochs 100

進階使用:
    python train_yolov8.py \
        --data dataset.yaml \
        --model yolov8s.pt \
        --epochs 300 \
        --batch 32 \
        --imgsz 1280 \
        --device 0 \
        --name my_experiment

恢復訓練:
    python train_yolov8.py --data dataset.yaml --resume
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')

    # 必要參數
    parser.add_argument('--data', type=str, required=True,
                        help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='model path (yolov8n.pt, yolov8s.pt, etc.)')

    # 訓練參數
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='image size (pixels)')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # 優化器參數
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--patience', type=int, default=50,
                        help='early stopping patience (epochs)')

    # 資料載入參數
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--cache', type=str, default='',
                        choices=['', 'ram', 'disk'],
                        help='cache images for faster training')

    # 輸出參數
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='project directory')
    parser.add_argument('--name', type=str, default='train',
                        help='experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='allow overwriting existing experiment')

    # 其他參數
    parser.add_argument('--resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='use pretrained weights')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='verbose output')

    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    # 載入模型
    if args.resume:
        # 恢復訓練
        model_path = f'{args.project}/{args.name}/weights/last.pt'
        if not Path(model_path).exists():
            print(f"❌ 找不到檢查點: {model_path}")
            print(f"   請確認路徑或移除 --resume 參數")
            return

        model = YOLO(model_path)
        print(f"📦 恢復訓練: {model_path}")
    else:
        # 新訓練
        model = YOLO(args.model)
        print(f"📦 載入模型: {args.model}")

    # 訓練配置
    print(f"\n⚙️  訓練配置:")
    print(f"   資料集: {args.data}")
    print(f"   模型: {args.model}")
    print(f"   訓練輪數: {args.epochs}")
    print(f"   批次大小: {args.batch}")
    print(f"   圖像大小: {args.imgsz}")
    print(f"   設備: {args.device}")
    print(f"   優化器: {args.optimizer}")
    print(f"   初始學習率: {args.lr0}")
    print(f"   工作線程: {args.workers}")

    # 開始訓練
    print(f"\n🚀 開始訓練...\n")
    results = model.train(
        # 基本參數
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        # 優化器參數
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,

        # 資料載入
        workers=args.workers,
        cache=args.cache if args.cache else False,

        # 輸出
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok or args.resume,

        # 其他
        pretrained=args.pretrained,
        verbose=args.verbose,
        save=True,
        save_period=10,
        plots=True,
    )

    # 訓練完成後驗證
    print(f"\n📊 驗證模型...")
    metrics = model.val()

    # 輸出結果
    print(f"\n{'='*60}")
    print(f"✅ 訓練完成！")
    print(f"{'='*60}")
    print(f"\n📈 性能指標:")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")

    print(f"\n💾 模型儲存位置:")
    print(f"   最佳模型: {args.project}/{args.name}/weights/best.pt")
    print(f"   最終模型: {args.project}/{args.name}/weights/last.pt")

    print(f"\n📊 訓練結果:")
    print(f"   結果目錄: {args.project}/{args.name}/")
    print(f"   訓練曲線: {args.project}/{args.name}/results.png")
    print(f"   混淆矩陣: {args.project}/{args.name}/confusion_matrix.png")

    print(f"\n🎯 下一步:")
    print(f"   1. 查看訓練曲線: {args.project}/{args.name}/results.png")
    print(f"   2. 使用模型推理: python inference.py --model {args.project}/{args.name}/weights/best.pt")
    print(f"   3. 導出模型: model.export(format='onnx')")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
