import os
import argparse
import shutil
from PIL import Image

def convert_yolo_seg_to_bbox(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(image_dir):
        if file.endswith(('.jpg', '.png', '.jpeg', '.tif')):
            img_path = os.path.join(image_dir, file)
            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception as e:
                print(f"Error opening image {file}: {e}")
                continue

            # Copy image to output images directory
            shutil.copy(img_path, os.path.join(images_dir, file))

            txt_file = os.path.splitext(file)[0] + '.txt'
            txt_path = os.path.join(label_dir, txt_file)
            bbox_lines = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 3:  # class_id + at least 3 points (6 values)
                            continue
                        try:
                            class_id = int(parts[0])
                            points = [float(p) for p in parts[1:]]
                        except ValueError:
                            continue

                        # Denormalize points
                        abs_points = []
                        for i in range(0, len(points), 2):
                            x = points[i] * w
                            y = points[i + 1] * h
                            abs_points.extend([x, y])

                        # Calculate bbox
                        xs = abs_points[::2]
                        ys = abs_points[1::2]
                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        # Normalize bbox for YOLO
                        x_center = (x_min + x_max) / 2 / w
                        y_center = (y_min + y_max) / 2 / h
                        width = (x_max - x_min) / w
                        height = (y_max - y_min) / h

                        bbox_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        bbox_lines.append(bbox_line)

            # Write bbox annotations to output labels directory
            output_txt_path = os.path.join(labels_dir, txt_file)
            with open(output_txt_path, 'w') as f:
                f.writelines(bbox_lines)

    print(f"Conversion complete. Output saved to {output_dir}")

image_dir = r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\testing_folder\split_dataset\test"
label_dir = r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\testing_folder\split_dataset\test"
output_dir = r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\converted_dataset\test"

convert_yolo_seg_to_bbox(image_dir, label_dir, output_dir)