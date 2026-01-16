#!/usr/bin/env python
# coding: utf-8

"""
VisDrone Object Tracking Pipeline
YOLOv8 Detection + BoT-SORT Tracker
"""

import cv2
import json
from pathlib import Path
from ultralytics import YOLO
import argparse

def create_video_from_images(image_dir, output_path, fps=30):
    """이미지 시퀀스를 비디오로 변환"""
    images = sorted(Path(image_dir).glob('*.jpg'))
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    
    frame = cv2.imread(str(images[0]))
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        out.write(frame)
    
    out.release()
    return output_path

def track_objects(video_path, model_path, output_dir, conf_threshold=0.25):
    """객체 추적 수행"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 출력 비디오 설정
    output_video = str(output_dir / 'tracking.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    # JSONL 로그 파일
    jsonl_path = output_dir / 'tracking_results.jsonl'
    jsonl_file = open(jsonl_path, 'w')
    
    frame_idx = 0
    
    # 추적 수행 (BoT-SORT tracker 사용)
    results = model.track(video_path, stream=True, conf=conf_threshold, 
                          tracker='botsort.yaml', persist=True)
    
    for result in results:
        frame = result.orig_img.copy()
        timestamp_ms = int(frame_idx * 1000 / fps)
        
        tracks = []
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            
            for box, track_id, cls, score in zip(boxes, track_ids, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                
                # 클래스 이름 매핑
                class_names = {0: 'bicycle', 1: 'people', 2: 'motor', 3: 'pedestrian'}
                cls_name = class_names.get(cls, f'class_{cls}')
                
                # 바운딩 박스 그리기
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # ID와 클래스 표시
                label = f'ID:{track_id} {cls_name} {score:.2f}'
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # JSONL 로그 데이터
                tracks.append({
                    'track_id': int(track_id),
                    'cls': cls_name,
                    'score': float(score),
                    'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # JSONL 로그 작성
        log_entry = {
            'frame_idx': frame_idx,
            'timestamp_ms': timestamp_ms,
            'tracks': tracks
        }
        jsonl_file.write(json.dumps(log_entry) + '\n')
        
        # 프레임 저장
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'Processed {frame_idx} frames...')
    
    cap.release()
    out.release()
    jsonl_file.close()
    
    print(f'\n✅ Tracking completed!')
    print(f'   Video: {output_video}')
    print(f'   JSONL: {jsonl_path}')
    print(f'   Total frames: {frame_idx}')

def main():
    parser = argparse.ArgumentParser(description='VisDrone Object Tracking')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, default='runs/detect/step1_imgsz1024/weights/best.pt',
                       help='Model path')
    parser.add_argument('--output', type=str, default='tracking_output',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    track_objects(args.video, args.model, args.output, args.conf)

if __name__ == '__main__':
    main()
