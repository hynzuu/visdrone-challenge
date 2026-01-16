#!/usr/bin/env python
# coding: utf-8

"""
Tracking 결과 분석 및 통계
"""

import json
from pathlib import Path
from collections import defaultdict
import argparse

def analyze_tracking_results(jsonl_path):
    """JSONL 로그 분석"""
    
    track_stats = defaultdict(lambda: {'frames': 0, 'class': None, 'avg_score': 0})
    class_counts = defaultdict(int)
    total_frames = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            total_frames += 1
            
            for track in data['tracks']:
                track_id = track['track_id']
                cls = track['cls']
                score = track['score']
                
                track_stats[track_id]['frames'] += 1
                track_stats[track_id]['class'] = cls
                track_stats[track_id]['avg_score'] += score
                
                class_counts[cls] += 1
    
    # 평균 스코어 계산
    for track_id in track_stats:
        frames = track_stats[track_id]['frames']
        track_stats[track_id]['avg_score'] /= frames
    
    print("=" * 60)
    print("TRACKING RESULTS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal Frames: {total_frames}")
    print(f"Unique Tracks: {len(track_stats)}")
    
    print("\n--- Class Distribution ---")
    for cls, count in sorted(class_counts.items()):
        print(f"{cls:15s}: {count:5d} detections")
    
    print("\n--- Top 10 Longest Tracks ---")
    sorted_tracks = sorted(track_stats.items(), 
                          key=lambda x: x[1]['frames'], 
                          reverse=True)[:10]
    
    for track_id, stats in sorted_tracks:
        print(f"ID {track_id:3d}: {stats['class']:12s} | "
              f"{stats['frames']:4d} frames | "
              f"avg_score: {stats['avg_score']:.3f}")
    
    print("\n--- Track Duration Statistics ---")
    durations = [stats['frames'] for stats in track_stats.values()]
    # 통계 계산
    if durations:
        print(f"Average duration: {sum(durations)/len(durations):.1f} frames")
        print(f"Max duration: {max(durations)} frames")
        print(f"Min duration: {min(durations)} frames")
    else:
        print("No tracks found")

def main():
    parser = argparse.ArgumentParser(description='Analyze tracking results')
    parser.add_argument('--jsonl', type=str, required=True,
                       help='JSONL log file path')
    
    args = parser.parse_args()
    
    analyze_tracking_results(args.jsonl)

if __name__ == '__main__':
    main()
