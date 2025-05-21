import os
import random
import shutil
import argparse
from pathlib import Path

def reduce_validation_folders(dataset_path, images_to_keep=5, backup=True):
    """
    Reduces the number of images in each category folder to the specified number.
    
    Args:
        dataset_path: Path to the validation dataset (containing category folders)
        images_to_keep: Number of images to keep in each folder
        backup: Whether to create backup files before deletion
    """
    # Get the absolute path for dataset
    dataset_path = os.path.abspath(dataset_path)
    print(f"Processing dataset at: {dataset_path}")
    
    # Create backup directory if required
    backup_dir = None
    if backup:
        backup_dir = os.path.join(os.path.dirname(dataset_path), "val_sorted_backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            print(f"Created backup directory: {backup_dir}")
    
    # Get all category folders
    category_folders = [f for f in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, f))]
    
    total_folders = len(category_folders)
    print(f"Found {total_folders} category folders")
    
    for idx, folder in enumerate(category_folders):
        folder_path = os.path.join(dataset_path, folder)
        images = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # If folder has more images than we want to keep
        if len(images) > images_to_keep:
            # Select random images to keep
            images_to_preserve = set(random.sample(images, images_to_keep))
            images_to_remove = [img for img in images if img not in images_to_preserve]
            
            # Create backup of files to be removed if requested
            if backup:
                category_backup_dir = os.path.join(backup_dir, folder)
                if not os.path.exists(category_backup_dir):
                    os.makedirs(category_backup_dir)
                
                for img in images_to_remove:
                    src_path = os.path.join(folder_path, img)
                    dst_path = os.path.join(category_backup_dir, img)
                    shutil.copy2(src_path, dst_path)
            
            # Remove extra images
            for img in images_to_remove:
                os.remove(os.path.join(folder_path, img))
            
            print(f"[{idx+1}/{total_folders}] Processed '{folder}': Kept {images_to_keep} images, removed {len(images_to_remove)} images")
        else:
            print(f"[{idx+1}/{total_folders}] Skipped '{folder}': Already has {len(images)} images (≤ {images_to_keep})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce the number of images in each category folder")
    parser.add_argument("--dataset", type=str, default="../imagenet/val_sorted", 
                        help="Path to the validation dataset")
    parser.add_argument("--keep", type=int, default=5, 
                        help="Number of images to keep in each folder")
    parser.add_argument("--no-backup", action="store_true", 
                        help="Don't create backups before removing files")
    
    args = parser.parse_args()
    
    reduce_validation_folders(args.dataset, args.keep, not args.no_backup)
    print("Done!")
