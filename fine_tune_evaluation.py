import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ---------------- Dataset ---------------- #
class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, processor, target_size=(792, 612)):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        self.processor = processor
        self.img_folder = img_folder
        self.images = {img['id']: img for img in self.coco['images']}
        self.target_size = target_size

        # Map categories
        category_ids = sorted(set(cat['id'] for cat in self.coco['categories']))
        self.category_map = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

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
        boxes = torch.tensor([ann["bbox"] for ann in annots], dtype=torch.float32)
        labels = torch.tensor([self.category_map[ann["category_id"]] for ann in annots], dtype=torch.int64)

        # Convert xywh -> xyxy (required by torchmetrics)
        if len(boxes) > 0:
            boxes[:, 2:] += boxes[:, :2]

        target = {"boxes": boxes, "labels": labels}

        # Process image with fixed size
        encoding = self.processor(images=image, return_tensors="pt", size=self.target_size)
        pixel_values = encoding["pixel_values"].squeeze()  # shape: [3,H,W]

        return pixel_values, target

# ---------------- Custom collate ---------------- #
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]  # list of tensors
    targets = [item[1] for item in batch]       # list of dicts
    pixel_values = torch.stack(pixel_values)
    return pixel_values, targets

# ---------------- Evaluation ---------------- #
def evaluate_map(model_path, img_dir, ann_file, batch_size=4, device="cuda"):
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    dataset = CocoDetection(img_dir, ann_file, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    metric = MeanAveragePrecision()
    model.eval()

    with torch.no_grad():
        for pixel_values, targets in dataloader:
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)

            # Post-process predictions
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=[dataset.target_size] * pixel_values.shape[0]
            )

            preds, gts = [], []
            for i, r in enumerate(results):
                boxes = r["boxes"].cpu()
                scores = r["scores"].cpu()
                labels = r["labels"].cpu()

                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": targets[i]["boxes"], "labels": targets[i]["labels"]})

            metric.update(preds, gts)

    final_results = metric.compute()
    print("\n---- Evaluation Results ----")
    for k, v in final_results.items():
        if torch.is_tensor(v):
            v = v.float().mean().item()  # average across classes
        print(f"{k}: {v:.4f}")

# ---------------- Run ---------------- #
if __name__ == "__main__":
    model_path = "./layout_detection_finetuned"
    test_img_dir = "/home/ubuntu/PS-5/Dataset/coco_dataset/test/images"
    test_ann_file = "/home/ubuntu/PS-5/Dataset/coco_dataset/test/annotations.json"

    evaluate_map(model_path, test_img_dir, test_ann_file, batch_size=4)
