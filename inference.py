import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection

# ---------------- Settings ---------------- #
model_path = "./layout_detection_finetuned"  # path to your trained DETR model
dataset_dir = "/home/ubuntu/PS-5/Dataset/mock_data_15_sept"  # path to dataset folder
output_dir = "/home/ubuntu/PS-5/mock_data_output"  # output folder

# Subdirectories for outputs
output_img_dir = os.path.join(output_dir, "output_images")
output_json_dir = os.path.join(output_dir, "output_json")
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

# Map class ids (0-based from model) to dataset ids + names
id2label = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure"
}

# ---------------- Load model ---------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = DetrImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path).to(device)
model.eval()

# ---------------- Inference ---------------- #
def run_inference(image_path, detection_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**encoding)

    # Post-process predictions
    target_size = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_size, threshold=detection_threshold
    )[0]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    annotations = []
    for bbox, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x_min, y_min, x_max, y_max = bbox.tolist()
        cls_id = int(label.item()) + 1  # make category_id 1-based
        cls_name = id2label.get(int(label.item()), "unknown")

        # Draw visualization
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.text((x_min, y_min - 10), f"{cls_name} ({score:.2f})", fill="red", font=font)

        # Convert to xyhw
        xyhw = [round(x_min, 2), round(y_min, 2), round(y_max - y_min, 2), round(x_max - x_min, 2)]

        annotations.append({
            "bbox": xyhw,
            "category_id": cls_id,
            "category_name": cls_name
        })

    # Final JSON format
    output_json = {
        "file_name": os.path.basename(image_path),
        "annotations": annotations,
        "corruption": {"type": "none", "severity": 0}
    }

    # Save JSON
    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    json_path = os.path.join(output_json_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    # Save visualized image
    vis_path = os.path.join(output_img_dir, f"{base_name}_vis.png")
    image.save(vis_path)

    print(f"✅ Processed {os.path.basename(image_path)} -> JSON + visualization saved.")


# ---------------- Batch Run ---------------- #
if __name__ == "__main__":
    image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"Found {len(image_files)} images in {dataset_dir}")
    for img_file in image_files:
        img_path = os.path.join(dataset_dir, img_file)
        run_inference(img_path, detection_threshold=0.5)

    print(f"\n🎯 All results saved to: {output_dir}")
