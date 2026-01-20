import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments

class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, processor):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        self.processor = processor
        self.img_folder = img_folder
        
        self.images = {img['id']: img for img in self.coco['images']}
        
        # Get unique category ids and map to consecutive indices starting from 0
        category_ids = sorted(set(cat['id'] for cat in self.coco['categories']))
        self.category_map = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        self.id2label = {idx: cat['name'] for cat in self.coco['categories'] for orig_id, idx in self.category_map.items() if orig_id == cat['id']}
        
        self.annots = {}
        for ann in self.coco['annotations']:
            if ann['image_id'] not in self.annots:
                self.annots[ann['image_id']] = []
            self.annots[ann['image_id']].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        annots = self.annots.get(img_id, [])
        
        target = {
            'image_id': torch.tensor([img_id]),
            'annotations': [
                {
                    'bbox': ann['bbox'],
                    'category_id': self.category_map[ann['category_id']],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd']
                } for ann in annots
            ]
        }
        
        encoding = self.processor(images=image, annotations=target, return_tensors="pt", size={"height": 792, "width": 612})
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_mask = encoding["pixel_mask"].squeeze()
        labels = encoding["labels"][0]
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels
        }

# Define collate function
def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch]
    pixel_mask = [item['pixel_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        "pixel_values": torch.stack(pixel_values),
        "pixel_mask": torch.stack(pixel_mask),
        "labels": labels
    }

# Set paths (adjust if necessary)
root_dir = "/home/ubuntu/PS-5/Dataset/coco_dataset"
train_img_dir = os.path.join(root_dir, "train", "images")
train_ann_file = os.path.join(root_dir, "train", "annotations.json")
val_img_dir = os.path.join(root_dir, "val", "images")
val_ann_file = os.path.join(root_dir, "val", "annotations.json")
test_img_dir = os.path.join(root_dir, "test", "images")
test_ann_file = os.path.join(root_dir, "test", "annotations.json")

# Load processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Adjust for new number of classes
num_labels = 5  
model.config.num_labels = num_labels

# Reinitialize the class classifier
import torch.nn as nn
model.class_labels_classifier = nn.Linear(model.config.d_model, num_labels + 1)  # +1 for 'no_object'

# Load datasets
train_dataset = CocoDetection(train_img_dir, train_ann_file, processor)
val_dataset = CocoDetection(val_img_dir, val_ann_file, processor)
test_dataset = CocoDetection(test_img_dir, test_ann_file, processor)

# Set id2label and label2id in model config
model.config.id2label = train_dataset.id2label
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

# Training arguments
training_args = TrainingArguments(
    output_dir="./layout_detection_finetuned",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_steps=1000,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=1e-4,
    remove_unused_columns=False,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("Test set evaluation results:", test_results)

# Save the model
trainer.save_model()
print("Model saved to ./layout_detection_finetuned")