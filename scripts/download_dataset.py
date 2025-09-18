"""
Script to download and prepare the Kaggle Car Plate Detection dataset.
"""
import os
import zipfile
import requests
from pathlib import Path
import shutil


def download_dataset(output_dir: str = "data/raw"):
    """
    Download the car plate detection dataset from Kaggle.
    
    Note: This requires Kaggle API credentials to be set up.
    See: https://github.com/Kaggle/kaggle-api#api-credentials
    
    Args:
        output_dir: Directory to save the dataset
    """
    try:
        import kaggle
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Downloading Kaggle Car Plate Detection dataset...")
        print("This may take a few minutes...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'andrewmvd/car-plate-detection',
            path=str(output_path),
            unzip=True
        )
        
        print(f"Dataset downloaded successfully to: {output_path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in output_path.rglob("*"):
            if file.is_file():
                print(f"  {file.relative_to(output_path)}")
        
        return str(output_path)
        
    except ImportError:
        print("Error: Kaggle API not installed. Install with: pip install kaggle")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure:")
        print("1. Kaggle API is installed: pip install kaggle")
        print("2. API credentials are set up: https://github.com/Kaggle/kaggle-api#api-credentials")
        return None


def download_sample_images():
    """
    Download sample license plate images for testing if Kaggle dataset is not available.
    """
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample license plate images...")
    
    # Sample URLs (placeholder - in practice, you'd use actual sample images)
    sample_urls = [
        # These would be actual URLs to sample license plate images
        # For now, we'll create placeholder files
    ]
    
    # Create some sample placeholder files
    sample_files = [
        "sample_plate_1.jpg",
        "sample_plate_2.jpg", 
        "sample_plate_3.jpg"
    ]
    
    for filename in sample_files:
        sample_path = sample_dir / filename
        # Create a small placeholder file
        with open(sample_path, 'w') as f:
            f.write(f"Placeholder for {filename}")
        print(f"Created placeholder: {sample_path}")
    
    print(f"\nSample files created in: {sample_dir}")
    print("Note: These are placeholder files. Replace with actual license plate images for testing.")
    
    return str(sample_dir)


def setup_directory_structure():
    """Create the standard directory structure for the project."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "models/pretrained",
        "models/trained",
        "results/images",
        "results/videos",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def create_sample_dataset_yaml():
    """Create a sample dataset YAML file for testing."""
    yaml_content = """
# License Plate Detection Dataset Configuration
path: data/processed  # Root directory
train: train/images   # Training images (relative to path)
val: val/images      # Validation images (relative to path)

# Classes
nc: 1  # number of classes
names: ['license_plate']  # class names

# Optional: Download script/URL (update with actual URL)
# download: https://github.com/your-repo/dataset-download-script.sh
"""
    
    yaml_path = Path("data/sample_dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"Sample dataset YAML created: {yaml_path}")
    return str(yaml_path)


def main():
    """Main function to set up the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and set up license plate dataset")
    parser.add_argument("--kaggle", action="store_true", 
                       help="Download from Kaggle (requires API setup)")
    parser.add_argument("--samples", action="store_true",
                       help="Create sample placeholder files for testing")
    parser.add_argument("--setup-dirs", action="store_true", 
                       help="Set up directory structure")
    parser.add_argument("--output-dir", default="data/raw",
                       help="Output directory for dataset")
    
    args = parser.parse_args()
    
    if args.setup_dirs or not any([args.kaggle, args.samples]):
        print("Setting up directory structure...")
        setup_directory_structure()
        create_sample_dataset_yaml()
    
    if args.kaggle:
        dataset_path = download_dataset(args.output_dir)
        if dataset_path:
            print(f"\n✅ Dataset ready at: {dataset_path}")
            print("\nNext steps:")
            print("1. Process the dataset: python src/data/dataset.py --input data/raw --output data/processed")
            print("2. Train the model: python src/training/train_detector.py --data data/processed/dataset.yaml")
    
    if args.samples:
        sample_path = download_sample_images()
        print(f"\n✅ Sample files created at: {sample_path}")
        print("\nNote: Replace placeholder files with actual license plate images for testing.")
    
    if not args.kaggle and not args.samples:
        print("\n📋 Directory structure created!")
        print("\nTo get started:")
        print("1. Download dataset: python scripts/download_dataset.py --kaggle")
        print("2. Or create samples: python scripts/download_dataset.py --samples") 
        print("3. Process data: python src/data/dataset.py --input data/raw --output data/processed")
        print("4. Train model: python src/training/train_detector.py --data data/processed/dataset.yaml")


if __name__ == "__main__":
    main()