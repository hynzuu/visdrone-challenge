#!/usr/bin/env python
# coding: utf-8

"""
VisDrone 이미지 시퀀스를 비디오로 변환
"""

import cv2
from pathlib import Path
import argparse

def images_to_video(image_dir, output_path, fps=30, target_size=(1920, 1080)):
    """이미지 시퀀스를 비디오로 변환"""
    images = sorted(Path(image_dir).glob('*.jpg'))
    
    if not images:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"Found {len(images)} images")
    
    # 목표 크기 설정
    w, h = target_size
    
    # AVI 포맷 + XVID 코덱 사용
    output_path = output_path.replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("❌ Failed to open video writer")
        return
    
    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            # 모든 이미지를 목표 크기로 리사이즈
            frame = cv2.resize(frame, (w, h))
            out.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} frames")
    
    out.release()
    print(f"\n✅ Video created: {output_path}")
    print(f"   Resolution: {w}x{h}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {len(images)}")

def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone images to video')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output', type=str, default='input_video.mp4',
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    
    args = parser.parse_args()
    
    images_to_video(args.input, args.output, args.fps)

if __name__ == '__main__':
    main()
