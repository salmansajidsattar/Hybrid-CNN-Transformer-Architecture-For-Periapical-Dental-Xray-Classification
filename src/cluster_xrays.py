
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from auto_cluster_periapical import AutomaticImageClusterer, manual_review_interface

def main():
    source_folder = Path("data/raw")
    target_folder=Path("data/processed")
    print("\n🚀 Starting automatic clustering...")
    print("   This may take a few minutes depending on number of images...")
    
    clusterer = AutomaticImageClusterer(source_folder, target_folder)
    stats = clusterer.run(visualize=True)
    
    if not stats:
        print("\n❌ Clustering failed!")
        return
    
    manual_review_interface(
            stats['periapical_dir'],
            stats['non_periapical_dir'],
            num_samples=10
        )
if __name__ == "__main__":
    main()