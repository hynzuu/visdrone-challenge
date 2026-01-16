#!/bin/bash

# OpenMP 충돌 방지
export KMP_DUPLICATE_LIB_OK=TRUE

# VisDrone Object Tracking Pipeline
# 전체 파이프라인을 한 번에 실행

echo "========================================="
echo "VisDrone Object Tracking Pipeline"
echo "========================================="

# 1. 비디오 생성 (validation 이미지 전체 사용)
echo ""
echo "[1/3] Creating video from validation images..."
python create_video.py \
  --input VisDrone2019-DET-val/images \
  --output input_video.avi \
  --fps 30

# 2. 객체 추적
echo ""
echo "[2/3] Running object tracking..."
python track.py \
  --video input_video.avi \
  --model runs/detect/step1_imgsz1024/weights/best.pt \
  --output tracking_output \
  --conf 0.5

# 3. 결과 분석
echo ""
echo "[3/3] Analyzing tracking results..."
python analyze_tracking.py \
  --jsonl tracking_output/tracking_results.jsonl

echo ""
echo "========================================="
echo "✅ Pipeline completed!"
echo "========================================="
echo "Output files:"
echo "  - tracking_output/tracking.mp4"
echo "  - tracking_output/tracking_results.jsonl"
