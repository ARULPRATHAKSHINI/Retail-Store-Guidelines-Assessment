import os
import shutil

# Create dataset directory structure
os.makedirs('dataset/compliant', exist_ok=True)      # Organized images here
os.makedirs('dataset/non-compliant', exist_ok=True)  # Cluttered images here

def organize_images(source_folder, target_folder):
    # Ensure source folder exists
    if not os.path.exists(source_folder):
        print(f"Warning: Source folder {source_folder} doesn't exist!")
        return
        
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Copy files
    for file in os.listdir(source_folder):
        src_path = os.path.join(source_folder, file)
        if os.path.isfile(src_path):
            dst_path = os.path.join(target_folder, file)
            shutil.copy(src_path, dst_path)

# Move ORGANIZED images to compliant folder (correct labeling)
organize_images(
    'simple_images/well_organized_retail_store_interior',  # Source of good images
    'dataset/compliant'                                    # Target for good images
)

# Move CLUTTERED images to non-compliant folder (correct labeling)
organize_images(
    'simple_images/cluttered_retail_store_interior',  # Source of bad images
    'dataset/non-compliant'                           # Target for bad images
)

print("Dataset organization completed with correct classes!")
print("Folder structure meaning:")
print("- 'compliant' folder: Contains ORGANIZED images")
print("- 'non-compliant' folder: Contains CLUTTERED images")