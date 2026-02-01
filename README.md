# VisDrone Object Detection: Model Comparison

## Problem Definition

### Dataset: VisDrone2019-DET

VisDrone is a challenging drone-based object detection dataset captured from aerial viewpoints at various altitudes and angles. This project focuses on detecting 4 visually similar classes that are particularly difficult for models to distinguish.

### Target Classes

We selected 4 classes that are small in size and visually ambiguous:

- **Person Group:**
  - `pedestrian`: People walking or standing
  - `people`: People sitting in vehicles or in non-standing poses

- **Two-wheeler Group:**
  - `bicycle`: Bicycles
  - `motor`: Motorcycles

### Key Challenges

#### 1. Small Object Size
Objects appear extremely small due to high drone altitude. When resized to standard input dimensions, critical shape information is lost, making detection nearly impossible.

#### 2. Visual Ambiguity
- **pedestrian vs people**: From aerial view, distinguishing standing vs sitting poses is highly ambiguous
- **bicycle vs motor**: Similar overall structure makes differentiation difficult

#### 3. Class Imbalance
- `pedestrian`: Abundant samples
- `people` and `bicycle`: Significantly fewer samples
- Models tend to be biased toward majority classes