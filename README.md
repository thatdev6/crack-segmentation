
# Crack Segmentation Using YOLOv8 and U-Net

This project explores the use of two powerful deep learning architectures—YOLOv8 and U-Net—for the task of crack detection and segmentation in infrastructure images. The goal is to compare their performance and demonstrate the effectiveness of each approach in detecting and segmenting surface cracks in concrete or road structures.

## Project Overview

- **Objective:** Detect and segment cracks in images using object detection (YOLOv8) and semantic segmentation (U-Net).
- **Tools & Frameworks:** PyTorch, OpenCV, Ultralytics YOLOv8, Keras
- **Input Data:** Images with visible surface cracks

## YOLOv8 for Crack Detection

YOLOv8 (You Only Look Once) is a state-of-the-art object detection model developed by Ultralytics. It is used here to perform bounding box-based crack detection.

- **Model:** YOLOv8n/v8s
- **Training Data:** Annotated images with bounding boxes around cracks
- **Output:** Bounding boxes indicating crack locations
- **Use Case:** Suitable for real-time detection and scenarios where crack localization is more important than pixel-wise segmentation

## U-Net for Crack Segmentation

U-Net is a convolutional neural network designed for biomedical image segmentation. It is used in this project to perform pixel-wise segmentation of cracks.

- **Architecture:** Encoder-decoder with skip connections
- **Input:** Crack images
- **Output:** Binary mask highlighting the exact location of cracks
- **Use Case:** High-resolution crack segmentation and detailed shape analysis

## Workflow

1. **Data Preparation**
   - Image resizing, normalization, and augmentation
   - Splitting into training, validation, and test sets

2. **Model Training**
   - YOLOv8 model is trained using the Ultralytics pipeline
   - U-Net model is trained using Keras/TensorFlow or PyTorch

3. **Evaluation**
   - Metrics: Accuracy, IoU (Intersection over Union), Precision, Recall, F1-Score
   - Visual comparison of prediction vs. ground truth

4. **Visualization**
   - Overlay of YOLOv8 detection boxes
   - U-Net segmentation masks on input images

## Results

The results compare the effectiveness of YOLOv8 for quick detection and U-Net for detailed segmentation. The final output demonstrates strengths and limitations of each model.

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow or Keras (for U-Net)
- OpenCV
- Ultralytics YOLOv8 (`pip install ultralytics`)
- NumPy, Matplotlib

## Conclusion

This project provides a practical comparison between object detection and semantic segmentation approaches for crack analysis, serving as a valuable tool for infrastructure maintenance and inspection tasks.

## License

This project is for educational and research purposes only.
