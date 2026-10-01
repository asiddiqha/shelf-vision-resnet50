import os
from PIL import Image
import numpy as np

# Define paths
base_dir = "shelf_dataset"
sub_dirs = ["train", "validation"]
classes = ["fully_stocked", "low_stock", "out_of_stock"]

# Create folders and dummy images
for sub in sub_dirs:
    for cls in classes:
        path = os.path.join(base_dir, sub, cls)
        os.makedirs(path, exist_ok=True)
        
        # Generate 5 dummy images per class to prevent FileNotFoundError
        for i in range(5):
            # Create a random color image array (224x224x3)
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(path, f"mock_img_{i}.jpg"))

print("✅ shelf_dataset folders and images created successfully!")
