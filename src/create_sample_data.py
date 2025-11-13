"""
Download sample e-commerce product images from Unsplash API.
Creates organized directory structure with product categories.
"""

import os
import requests
import json
from pathlib import Path
from typing import List, Dict
import time

# Configuration
UNSPLASH_API_KEY = "YOUR_UNSPLASH_API_KEY"  # Get free key from: https://unsplash.com/oauth/applications
DATASET_DIR = Path("data/raw/sample_products")
CATEGORIES = {
    "shoes": ["running shoes", "dress shoes", "sneakers", "boots"],
    "bags": ["backpack", "handbag", "crossbody bag", "tote bag"],
    "shirts": ["t-shirt", "polo shirt", "dress shirt", "hoodie"],
    "electronics": ["laptop", "smartphone", "headphones", "smartwatch"]
}
IMAGES_PER_CATEGORY = 5  # Download 5 images per subcategory


def create_directory_structure() -> Dict[str, Path]:
    """Create organized directory structure for product images."""
    category_dirs = {}
    for category in CATEGORIES.keys():
        category_path = DATASET_DIR / category
        category_path.mkdir(parents=True, exist_ok=True)
        category_dirs[category] = category_path
        print(f"✓ Created directory: {category_path}")
    return category_dirs


def download_images_from_unsplash() -> bool:
    """
    Download product images from Unsplash.
    Note: Requires API key from https://unsplash.com/oauth/applications
    """
    if UNSPLASH_API_KEY == "YOUR_UNSPLASH_API_KEY":
        print("\n⚠️  Unsplash API key not configured.")
        print("Get a free API key from: https://unsplash.com/oauth/applications")
        print("Then update UNSPLASH_API_KEY in this script.")
        return False
    
    category_dirs = create_directory_structure()
    total_downloaded = 0
    
    for category, subcategories in CATEGORIES.items():
        for subcategory in subcategories:
            search_query = f"{subcategory} product"
            url = "https://api.unsplash.com/search/photos"
            params = {
                "query": search_query,
                "per_page": IMAGES_PER_CATEGORY,
                "client_id": UNSPLASH_API_KEY
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for idx, photo in enumerate(data.get("results", [])):
                    img_url = photo["urls"]["regular"]
                    img_response = requests.get(img_url)
                    img_response.raise_for_status()
                    
                    # Save image
                    filename = f"{subcategory.replace(' ', '_')}_{idx}.jpg"
                    filepath = category_dirs[category] / filename
                    
                    with open(filepath, "wb") as f:
                        f.write(img_response.content)
                    
                    print(f"✓ Downloaded: {filepath}")
                    total_downloaded += 1
                    time.sleep(0.5)  # Rate limiting
                    
            except Exception as e:
                print(f"✗ Error downloading {search_query}: {e}")
    
    return total_downloaded > 0


def create_sample_images_local() -> bool:
    """
    Create sample images locally using PIL.
    Alternative to Unsplash when API key is not available.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("⚠️  PIL not installed. Install with: pip install Pillow")
        return False
    
    category_dirs = create_directory_structure()
    colors = {
        "shoes": "#FF6B6B",
        "bags": "#4ECDC4",
        "shirts": "#45B7D1",
        "electronics": "#FFA07A"
    }
    
    image_count = 0
    for category, subcategories in CATEGORIES.items():
        for idx, subcategory in enumerate(subcategories):
            # Create a simple colored image with text
            img = Image.new("RGB", (400, 400), color=colors[category])
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                # Try to use default font
                draw.text(
                    (200, 150),
                    subcategory.upper(),
                    fill="white",
                    anchor="mm"
                )
                draw.text(
                    (200, 250),
                    f"Sample {category}",
                    fill="white",
                    anchor="mm"
                )
            except:
                # Fallback if font fails
                draw.text((50, 150), subcategory.upper(), fill="white")
                draw.text((50, 250), f"Sample {category}", fill="white")
            
            # Save images (create 3 variations per subcategory)
            for variant in range(3):
                filename = f"{subcategory.replace(' ', '_')}_{variant}.png"
                filepath = category_dirs[category] / filename
                img.save(filepath)
                print(f"✓ Created: {filepath}")
                image_count += 1
    
    return image_count > 0


def create_metadata_file(category_dirs: Dict[str, Path]) -> bool:
    """Create metadata JSON file with image categorization."""
    metadata = {"categories": {}}
    
    for category, category_path in category_dirs.items():
        metadata["categories"][category] = {
            "path": str(category_path),
            "images": []
        }
        
        # List all images in category
        for img_file in category_path.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                metadata["categories"][category]["images"].append(img_file.name)
    
    metadata_file = DATASET_DIR / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to: {metadata_file}")
    return True


def print_summary(category_dirs: Dict[str, Path]) -> None:
    """Print summary of created dataset."""
    print("\n" + "="*60)
    print("SAMPLE DATASET CREATED")
    print("="*60)
    
    total_images = 0
    for category, category_path in category_dirs.items():
        images = list(category_path.glob("*"))
        count = len([f for f in images if f.is_file()])
        total_images += count
        print(f"\n{category.upper()}: {count} images")
        for img in sorted(images)[:3]:  # Show first 3
            print(f"  - {img.name}")
        if count > 3:
            print(f"  ... and {count - 3} more")
    
    print(f"\nTotal images: {total_images}")
    print(f"Dataset location: {DATASET_DIR}")
    print("="*60)


def main():
    """Main function to set up sample data."""
    print("Sample Product Image Dataset Generator")
    print("="*60)
    
    # Try to download from Unsplash first
    if UNSPLASH_API_KEY != "YOUR_UNSPLASH_API_KEY":
        print("\nAttempting to download from Unsplash...")
        if download_images_from_unsplash():
            category_dirs = {cat: DATASET_DIR / cat for cat in CATEGORIES.keys()}
            create_metadata_file(category_dirs)
            print_summary(category_dirs)
            return
        else:
            print("\n⚠️  Unsplash download failed. Falling back to local image creation...\n")
    
    # Fall back to creating sample images locally
    print("Creating sample images locally with PIL...")
    if create_sample_images_local():
        category_dirs = {cat: DATASET_DIR / cat for cat in CATEGORIES.keys()}
        create_metadata_file(category_dirs)
        print_summary(category_dirs)
    else:
        print("\n✗ Failed to create sample images. Please install Pillow: pip install Pillow")


if __name__ == "__main__":
    main()
