# DETR Fine-Tuning for Document Layout Detection

A complete pipeline for fine-tuning Facebook's DETR (DEtection TRansformer) model for document layout analysis. This project includes dataset preparation, model training, evaluation, and inference tools with support for multilingual documents.

## Overview

This repository provides tools to fine-tune a DETR model for detecting document layout elements such as text blocks, titles, lists, tables, and figures. The pipeline is designed for processing multilingual documents with support for English, Arabic, Hindi, Nepali, and Persian.

## Features

- **COCO Format Conversion**: Convert custom annotations to COCO format with automatic train/val/test splits
- **DETR Fine-Tuning**: Fine-tune `facebook/detr-resnet-50` for custom object detection tasks
- **mAP Evaluation**: Comprehensive evaluation using torchmetrics Mean Average Precision
- **Batch Inference**: Process multiple images with visualization and JSON output
- **Image Preprocessing**: Automatic deskewing and orientation correction using OCR-based scoring
- **Multilingual Support**: OCR-based preprocessing for documents in multiple languages

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.10+

### Install Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install torchmetrics
pip install pillow
pip install opencv-python
pip install scipy
pip install pytesseract  # For multilingual OCR preprocessing
```

For OCR-based preprocessing (optional):
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ara tesseract-ocr-hin tesseract-ocr-nep tesseract-ocr-fas

# macOS
brew install tesseract
brew install tesseract-lang
```

## Dataset Preparation

### Input Format

Your dataset should have JSON annotation files with the following structure:

```json
{
  "file_name": "image001.png",
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1
    }
  ]
}
```

### Convert to COCO Format

Edit the paths in `coco_conversion.py` and run:

```bash
python coco_conversion.py
```

This will:
1. Split data into train (80%), validation (10%), and test (10%) sets
2. Create COCO-format annotation files
3. Organize images into separate folders

**Output Structure:**
```
coco_dataset/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

## Training

### Configure Training

Edit paths in `model_split_train.py`:

```python
root_dir = "/path/to/coco_dataset"
num_labels = 5  # Number of classes
```

### Training Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 50 | Number of training epochs |
| `per_device_train_batch_size` | 8 | Batch size for training |
| `per_device_eval_batch_size` | 4 | Batch size for evaluation |
| `learning_rate` | 5e-5 | Learning rate |
| `weight_decay` | 1e-4 | Weight decay |
| `fp16` | True | Mixed precision training |

### Run Training

```bash
python model_split_train.py
```

The fine-tuned model will be saved to `./layout_detection_finetuned/`.

## Evaluation

Evaluate your trained model using Mean Average Precision (mAP):

```python
from fine_tune_evaluation import evaluate_map

evaluate_map(
    model_path="./layout_detection_finetuned",
    img_dir="/path/to/test/images",
    ann_file="/path/to/test/annotations.json",
    batch_size=4,
    device="cuda"
)
```

**Metrics Reported:**
- mAP (Mean Average Precision)
- mAP@50 (mAP at IoU=0.50)
- mAP@75 (mAP at IoU=0.75)
- mAP per class

## Inference

### Basic Inference

Edit paths in `inference.py` and run:

```bash
python inference.py
```

### Output

For each input image, the script generates:
- **Visualization**: Image with bounding boxes and class labels
- **JSON annotations**: Detected objects with coordinates and confidence scores

```json
{
  "file_name": "document.png",
  "annotations": [
    {
      "bbox": [x, y, height, width],
      "category_id": 1,
      "category_name": "text"
    }
  ],
  "corruption": {"type": "none", "severity": 0}
}
```

### Advanced Inference (with Preprocessing)

Use `training.ipynb` for inference with automatic:
- **Deskewing**: Corrects rotated documents
- **Orientation Detection**: Finds the best orientation (0°, 90°, 180°, 270°)
- **Flip Detection**: Detects horizontal/vertical flips

## Classes

The model is trained to detect 5 document layout classes:

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | text | Body text paragraphs |
| 1 | title | Headings and titles |
| 2 | list | Bulleted or numbered lists |
| 3 | table | Tables and tabular data |
| 4 | figure | Images, charts, diagrams |

## ⚙️ Configuration

### Model Configuration

```python
# Image size for processing
target_size = (792, 612)  # height, width

# Detection threshold
detection_threshold = 0.5
```

### Modifying Classes

To train with different classes:

1. Update `num_labels` in training script
2. Update `id2label` mapping in inference scripts
3. Ensure your annotations use the correct `category_id` values

## Usage Examples

### Quick Start

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

# Load model
model_path = "./layout_detection_finetuned"
processor = DetrImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path)

# Run inference
image = Image.open("document.png")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process
target_size = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_size, threshold=0.5
)[0]

# Print detections
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score:.2f}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Facebook DETR](https://github.com/facebookresearch/detr) - DEtection TRansformer
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model implementations
- [TorchMetrics](https://torchmetrics.readthedocs.io/) - Evaluation metrics
