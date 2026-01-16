#!/usr/bin/env python
# coding: utf-8

"""
Confidence Score 분석 및 최적 Threshold 결정
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def analyze_confidence_scores(jsonl_path):
    """JSONL에서 confidence score 분석"""
    
    scores_by_class = defaultdict(list)
    all_scores = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for track in data['tracks']:
                score = track['score']
                cls = track['cls']
                scores_by_class[cls].append(score)
                all_scores.append(score)
    
    return scores_by_class, all_scores

def plot_confidence_distribution(scores_by_class, all_scores):
    """Confidence score 분포 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 전체 분포 히스토그램
    ax = axes[0, 0]
    ax.hist(all_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(all_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_scores):.3f}')
    ax.axvline(np.median(all_scores), color='green', linestyle='--',
               label=f'Median: {np.median(all_scores):.3f}')
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Overall Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 클래스별 분포
    ax = axes[0, 1]
    colors = {'bicycle': '#FF6B6B', 'people': '#4ECDC4', 
              'motor': '#45B7D1', 'pedestrian': '#FFA07A'}
    
    for cls, scores in scores_by_class.items():
        ax.hist(scores, bins=30, alpha=0.5, label=cls, 
                color=colors.get(cls, 'gray'), edgecolor='black')
    
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Score by Class', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. 누적 분포 (CDF)
    ax = axes[1, 0]
    sorted_scores = np.sort(all_scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, linewidth=2)
    
    # 주요 threshold 표시
    thresholds = [0.25, 0.3, 0.4, 0.5]
    for thresh in thresholds:
        kept = np.sum(np.array(all_scores) >= thresh) / len(all_scores) * 100
        ax.axvline(thresh, linestyle='--', alpha=0.7, 
                   label=f'{thresh:.2f} ({kept:.1f}% kept)')
    
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Box plot (클래스별)
    ax = axes[1, 1]
    data = [scores_by_class[cls] for cls in sorted(scores_by_class.keys())]
    labels = sorted(scores_by_class.keys())
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, cls in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(cls, 'gray'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Score Distribution by Class', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('images/confidence_score_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: images/confidence_score_analysis.png")
    plt.show()

def recommend_threshold(scores_by_class, all_scores):
    """최적 threshold 추천"""
    
    print("\n" + "="*60)
    print("CONFIDENCE SCORE ANALYSIS")
    print("="*60)
    
    print(f"\nTotal detections: {len(all_scores)}")
    print(f"Mean score: {np.mean(all_scores):.3f}")
    print(f"Median score: {np.median(all_scores):.3f}")
    print(f"Std dev: {np.std(all_scores):.3f}")
    
    print("\n--- Class Statistics ---")
    for cls in sorted(scores_by_class.keys()):
        scores = scores_by_class[cls]
        print(f"{cls:12s}: mean={np.mean(scores):.3f}, "
              f"median={np.median(scores):.3f}, "
              f"min={np.min(scores):.3f}, "
              f"count={len(scores)}")
    
    print("\n--- Threshold Impact ---")
    thresholds = [0.25, 0.3, 0.35, 0.4, 0.5]
    
    for thresh in thresholds:
        kept = np.sum(np.array(all_scores) >= thresh)
        pct = kept / len(all_scores) * 100
        
        kept_by_class = {}
        for cls, scores in scores_by_class.items():
            kept_cls = np.sum(np.array(scores) >= thresh)
            kept_by_class[cls] = kept_cls
        
        print(f"\nThreshold {thresh:.2f}:")
        print(f"  Total kept: {kept}/{len(all_scores)} ({pct:.1f}%)")
        for cls in sorted(kept_by_class.keys()):
            total = len(scores_by_class[cls])
            kept_cls = kept_by_class[cls]
            pct_cls = kept_cls / total * 100 if total > 0 else 0
            print(f"  {cls:12s}: {kept_cls}/{total} ({pct_cls:.1f}%)")
    
    # 추천
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    # 중앙값과 평균 사이 값 추천
    median = np.median(all_scores)
    mean = np.mean(all_scores)
    
    # 0.05 단위로 반올림
    recommended = round(median / 0.05) * 0.05
    
    # 0.3~0.5 범위로 제한
    recommended = max(0.3, min(0.5, recommended))
    
    print(f"\nRecommended threshold: {recommended:.2f}")
    print(f"  Median score: {median:.3f}")
    print(f"  Mean score: {mean:.3f}")
    
    kept = np.sum(np.array(all_scores) >= recommended)
    pct = kept / len(all_scores) * 100
    print(f"  Will keep: {kept}/{len(all_scores)} ({pct:.1f}%) detections")
    
    return recommended

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze confidence scores')
    parser.add_argument('--jsonl', type=str, required=True,
                       help='JSONL log file path')
    
    args = parser.parse_args()
    
    scores_by_class, all_scores = analyze_confidence_scores(args.jsonl)
    plot_confidence_distribution(scores_by_class, all_scores)
    recommended = recommend_threshold(scores_by_class, all_scores)

if __name__ == '__main__':
    main()
