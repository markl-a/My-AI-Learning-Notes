#!/usr/bin/env python3
"""
YOLOv8 模型評估腳本

使用方式:
    # 在驗證集上評估
    python evaluate.py --model best.pt --data dataset.yaml

    # 在測試集上評估
    python evaluate.py --model best.pt --data dataset.yaml --split test

    # 自定義評估參數
    python evaluate.py --model best.pt --data dataset.yaml --imgsz 1280 --batch 8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='YOLOv8 Model Evaluation')

    # 必要參數
    parser.add_argument('--model', type=str, required=True,
                        help='model path (e.g., best.pt)')
    parser.add_argument('--data', type=str, required=True,
                        help='dataset.yaml path')

    # 評估參數
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='dataset split to evaluate')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='inference image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='NMS IoU threshold')

    # 輸出參數
    parser.add_argument('--save-json', action='store_true',
                        help='save results to JSON')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='save hybrid version of labels')
    parser.add_argument('--project', type=str, default='runs/val',
                        help='save results to project/name')
    parser.add_argument('--name', type=str, default='eval',
                        help='save results to project/name')

    # 其他參數
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='verbose output')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='generate plots')

    return parser.parse_args()


def print_metrics(metrics, model):
    """列印評估指標"""
    print(f"\n{'='*60}")
    print(f"📊 整體性能指標")
    print(f"{'='*60}\n")

    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"mAP75:      {metrics.box.map75:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

    # 每類別的性能
    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
        print(f"\n{'='*60}")
        print(f"📈 各類別性能 (mAP50-95)")
        print(f"{'='*60}\n")

        class_names = model.names
        for i, map_value in enumerate(metrics.box.maps):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"{class_name:15s}: {map_value:.4f}")

    print(f"\n{'='*60}\n")


def plot_class_performance(metrics, model, save_path):
    """繪製各類別性能圖表"""
    if not hasattr(metrics.box, 'maps') or len(metrics.box.maps) == 0:
        return

    class_names = [model.names[i] for i in range(len(metrics.box.maps))]
    maps = metrics.box.maps

    plt.figure(figsize=(12, 6))

    # mAP 柱狀圖
    plt.subplot(1, 2, 1)
    plt.bar(range(len(maps)), maps, color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('mAP50-95')
    plt.title('mAP per Class')
    plt.xticks(range(len(maps)), class_names, rotation=45, ha='right')
    plt.tight_layout()

    # 性能雷達圖
    if len(maps) >= 3:
        plt.subplot(1, 2, 2, projection='polar')
        angles = [n / len(maps) * 2 * 3.14159 for n in range(len(maps))]
        maps_plot = list(maps) + [maps[0]]
        angles_plot = angles + [angles[0]]

        plt.plot(angles_plot, maps_plot, 'o-', linewidth=2, color='steelblue')
        plt.fill(angles_plot, maps_plot, alpha=0.25, color='steelblue')
        plt.xticks(angles, class_names)
        plt.ylim(0, 1)
        plt.title('Performance Radar Chart')

    plt.tight_layout()
    plt.savefig(save_path / 'class_performance.png', dpi=300, bbox_inches='tight')
    print(f"📊 類別性能圖表已儲存: {save_path / 'class_performance.png'}")


def save_metrics_to_csv(metrics, model, save_path):
    """儲存指標到 CSV"""
    data = {
        'Metric': ['mAP50', 'mAP50-95', 'mAP75', 'Precision', 'Recall'],
        'Value': [
            metrics.box.map50,
            metrics.box.map,
            metrics.box.map75,
            metrics.box.mp,
            metrics.box.mr,
        ]
    }

    df = pd.DataFrame(data)
    csv_path = save_path / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"📊 整體指標已儲存: {csv_path}")

    # 各類別指標
    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
        class_data = {
            'Class': [model.names[i] for i in range(len(metrics.box.maps))],
            'mAP50-95': metrics.box.maps,
        }
        class_df = pd.DataFrame(class_data)
        class_csv_path = save_path / 'class_metrics.csv'
        class_df.to_csv(class_csv_path, index=False)
        print(f"📊 類別指標已儲存: {class_csv_path}")


def main():
    """主函數"""
    args = parse_args()

    # 載入模型
    print(f"📦 載入模型: {args.model}")
    model = YOLO(args.model)

    # 顯示配置
    print(f"\n⚙️  評估配置:")
    print(f"   資料集: {args.data}")
    print(f"   劃分: {args.split}")
    print(f"   圖像大小: {args.imgsz}")
    print(f"   批次大小: {args.batch}")
    print(f"   信心閾值: {args.conf}")
    print(f"   IoU 閾值: {args.iou}")

    # 執行評估
    print(f"\n🚀 開始評估...\n")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        project=args.project,
        name=args.name,
        verbose=args.verbose,
        plots=args.plots,
    )

    # 顯示結果
    print_metrics(metrics, model)

    # 建立輸出目錄
    save_path = Path(args.project) / args.name
    save_path.mkdir(parents=True, exist_ok=True)

    # 儲存指標
    save_metrics_to_csv(metrics, model, save_path)

    # 繪製圖表
    if args.plots:
        plot_class_performance(metrics, model, save_path)

    # 總結
    print(f"✅ 評估完成！")
    print(f"\n💾 結果已儲存:")
    print(f"   目錄: {save_path}/")
    if args.plots:
        print(f"   混淆矩陣: {save_path}/confusion_matrix.png")
        print(f"   F1 曲線: {save_path}/F1_curve.png")
        print(f"   PR 曲線: {save_path}/PR_curve.png")
        print(f"   類別性能: {save_path}/class_performance.png")
    print(f"   指標 CSV: {save_path}/metrics.csv")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
