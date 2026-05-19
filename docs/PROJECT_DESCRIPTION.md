# Project Description

## Title

Cattle Monitoring System Using YOLOv8 Detection and Instance Segmentation

## Problem

Manual cattle monitoring from camera footage is time-consuming. A useful monitoring system should estimate how many cattle are present, how much area they occupy, and when crowded or sudden-change events happen.

## Approach

The project trains and compares two YOLOv8 models:

1. YOLOv8s object detection
2. YOLOv8s instance segmentation

Detection gives cattle bounding boxes. Segmentation gives cattle masks. After prediction, model outputs are converted into practical monitoring signals.

## Monitoring Signals

The system computes:

- frame-level cattle count
- detection box occupancy
- segmentation mask occupancy
- density status
- occupancy status
- event status

## Events

The system detects:

- EMPTY_SCENE
- HIGH_DENSITY
- HIGH_OCCUPANCY
- SUDDEN_COUNT_INCREASE
- SUDDEN_COUNT_DECREASE

## Final Finding

Segmentation outperformed detection in both count estimation and occupancy estimation. It achieved lower count MAE and much lower occupancy MAE, making it more suitable for cattle monitoring tasks where spatial coverage matters.
