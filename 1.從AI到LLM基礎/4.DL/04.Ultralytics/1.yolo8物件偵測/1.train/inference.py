#!/usr/bin/env python3
"""
YOLOv8 推理腳本

使用方式:
    # 單張圖像
    python inference.py --model best.pt --source image.jpg

    # 多張圖像
    python inference.py --model best.pt --source images/

    # 影片
    python inference.py --model best.pt --source video.mp4

    # 網路攝影機
    python inference.py --model best.pt --source 0

    # YouTube 影片
    python inference.py --model best.pt --source https://youtube.com/watch?v=...
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')

    # 必要參數
    parser.add_argument('--model', type=str, required=True,
                        help='model path (e.g., best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='image/video/directory/webcam (0 for default camera)')

    # 推理參數
    parser.add_argument('--conf', type=float, default=0.25,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='inference image size')
    parser.add_argument('--max-det', type=int, default=300,
                        help='maximum detections per image')

    # 過濾參數
    parser.add_argument('--classes', type=int, nargs='+',
                        help='filter by class (e.g., --classes 0 1 2)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or cpu')

    # 輸出參數
    parser.add_argument('--save', action='store_true', default=True,
                        help='save results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', type=str, default='predict',
                        help='save results to project/name')

    # 顯示參數
    parser.add_argument('--show', action='store_true',
                        help='show results')
    parser.add_argument('--show-labels', action='store_true', default=True,
                        help='show labels')
    parser.add_argument('--show-conf', action='store_true', default=True,
                        help='show confidences')
    parser.add_argument('--line-width', type=int, default=2,
                        help='bounding box line width')

    # 進階參數
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--stream-buffer', action='store_true',
                        help='buffer all streaming frames (True) or return the most recent frame (False)')

    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    # 載入模型
    print(f"📦 載入模型: {args.model}")
    model = YOLO(args.model)

    # 顯示配置
    print(f"\n⚙️  推理配置:")
    print(f"   來源: {args.source}")
    print(f"   信心閾值: {args.conf}")
    print(f"   IoU 閾值: {args.iou}")
    print(f"   圖像大小: {args.imgsz}")
    print(f"   最大檢測數: {args.max_det}")
    if args.classes:
        print(f"   過濾類別: {args.classes}")

    # 執行推理
    print(f"\n🚀 開始推理...\n")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        classes=args.classes,
        device=args.device,

        # 輸出選項
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,

        # 顯示選項
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_width,

        # 進階選項
        vid_stride=args.vid_stride,
        stream_buffer=args.stream_buffer,
    )

    # 處理結果
    total_detections = 0
    for i, result in enumerate(results):
        num_boxes = len(result.boxes)
        total_detections += num_boxes

        if num_boxes > 0:
            print(f"\n圖像 {i+1}:")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  - {class_name}: {conf:.2f}")

    # 總結
    print(f"\n{'='*60}")
    print(f"✅ 推理完成！")
    print(f"{'='*60}")
    print(f"\n📊 統計:")
    print(f"   處理圖像數: {len(results)}")
    print(f"   總檢測數: {total_detections}")
    print(f"   平均每張: {total_detections/len(results):.1f}")

    if args.save:
        print(f"\n💾 結果已儲存:")
        print(f"   位置: {args.project}/{args.name}/")
        if args.save_txt:
            print(f"   標籤檔案: {args.project}/{args.name}/labels/")
        if args.save_crop:
            print(f"   裁剪圖像: {args.project}/{args.name}/crops/")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
