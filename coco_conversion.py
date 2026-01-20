import os
import json
import random
import shutil
from PIL import Image

# Input folders
json_folder = r"/home/ubuntu/PS-5/Dataset/train"
img_folder = r"/home/ubuntu/PS-5/Dataset/train"
out_dir = r"/home/ubuntu/PS-5/Dataset/coco_dataset"
os.makedirs(out_dir, exist_ok=True)

# Split ratios
split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test

# Collect all JSONs
all_data = []
for json_file in os.listdir(json_folder):
    if not json_file.endswith(".json"):
        continue
    json_path = os.path.join(json_folder, json_file)
    with open(json_path, "r") as f:
        all_data.append(json.load(f))

# Shuffle for random split
random.shuffle(all_data)
n = len(all_data)
train_end = int(split_ratio[0] * n)
val_end = train_end + int(split_ratio[1] * n)

splits = {
    "train": all_data[:train_end],
    "val": all_data[train_end:val_end],
    "test": all_data[val_end:]
}

def build_coco(subset, split_name):
    coco = {"images": [], "annotations": [], "categories": []}
    categories = {}
    ann_id = 1
    img_id = 1

    # Create folders for split
    split_dir = os.path.join(out_dir, split_name)
    img_out_dir = os.path.join(split_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)

    for data in subset:
        img_path = os.path.join(img_folder, data["file_name"])
        if not os.path.exists(img_path):
            print(f"⚠️ Missing image {data['file_name']}, skipping...")
            continue

        # Copy image to split folder
        shutil.copy(img_path, os.path.join(img_out_dir, data["file_name"]))

        # Image size
        with Image.open(img_path) as img:
            W, H = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": data["file_name"],
            "width": W,
            "height": H
        })

        for ann in data["annotations"]:
            cid = ann["category_id"]
            if cid not in categories:
                categories[cid] = f"class_{cid}"

            x, y, w, h = ann["bbox"]

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco["categories"] = [{"id": cid, "name": name} for cid, name in categories.items()]

    # Save annotations JSON inside split folder
    out_file = os.path.join(split_dir, "annotations.json")
    with open(out_file, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"✅ {split_name}: {len(coco['images'])} images saved with annotations")

# Build each split
for split_name, subset in splits.items():
    build_coco(subset, split_name)

print("\n🎉 Dataset ready in COCO format at:", out_dir)