
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class AutomaticImageClusterer:
    def __init__(self, source_folder, target_folder=None):
        self.source_folder = Path(source_folder)
        self.target_folder = Path(target_folder)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.periapical_dir = self.target_folder / "periapical"
        self.non_periapical_dir = self.target_folder / "non_periapical"
        
        self.periapical_dir.mkdir(parents=True, exist_ok=True)
        self.non_periapical_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Loading feature extractor (ResNet50)...")
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        print("✓ Feature extractor loaded")
    
    def find_all_images(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        print(f"\nScanning folder: {self.source_folder}")
        for ext in image_extensions:
            image_paths.extend(list(self.source_folder.glob(ext)))
            image_paths.extend(list(self.source_folder.glob(f"**/{ext}")))
        
        image_paths = list(set(image_paths))
        
        print(f"Found {len(image_paths)} images")
        return image_paths
    
    def extract_features(self, image_paths):
        features_list = []
        valid_paths = []
        
        print("\nExtracting  images")
        
        for img_path in tqdm(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.feature_extractor(image_tensor)
                
                features_list.append(features.cpu().numpy().flatten())
                valid_paths.append(img_path)
                
            except Exception as e:
                print(f"\n Not Found {img_path}: {e}")
                continue
        
        features = np.array(features_list)
        return features, valid_paths
    
    def cluster_images(self, features, n_clusters=2):
        pca = PCA(n_components=min(50, features.shape[1]))
        features_pca = pca.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
    
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count} images ({count/len(labels)*100:.1f}%)")    
        return labels, kmeans, pca
    
    def identify_periapical_cluster(self, features, labels, image_paths):
        cluster_stats = {}
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_image_paths = [path for path, mask in zip(image_paths, cluster_mask) if mask]
            
            aspect_ratios = []
            intensities = []
            widths = []
            
            sample_paths = np.random.choice(
                cluster_image_paths, 
                size=min(50, len(cluster_image_paths)), 
                replace=False
            )
            
            for img_path in sample_paths:
                try:
                    img = Image.open(img_path).convert('L')
                    width, height = img.size
                    aspect_ratio = height / width
                    
                    img_array = np.array(img)
                    intensity = img_array.mean()
                    
                    aspect_ratios.append(aspect_ratio)
                    intensities.append(intensity)
                    widths.append(width)
                except:
                    continue
            
            cluster_stats[cluster_id] = {
                'aspect_ratio_mean': np.mean(aspect_ratios),
                'aspect_ratio_std': np.std(aspect_ratios),
                'intensity_mean': np.mean(intensities),
                'intensity_std': np.std(intensities),
                'width_mean': np.mean(widths),
                'count': sum(cluster_mask)
            }
        
        print("\nCluster Statistics:")
        print(f"{'Cluster':<10} {'Count':<10} {'Aspect Ratio':<20} {'Intensity':<20} {'Width':<15}")
        print("-" * 75)
        for cluster_id, stats in cluster_stats.items():
            print(f"{cluster_id:<10} {stats['count']:<10} "
                f"{stats['aspect_ratio_mean']:.2f}±{stats['aspect_ratio_std']:.2f}{'':<12} "
                f"{stats['intensity_mean']:.1f}±{stats['intensity_std']:.1f}{'':<8} "
                f"{stats['width_mean']:.0f}")
        
        periapical_cluster_id = min(
            cluster_stats.items(), 
            key=lambda x: x[1]['aspect_ratio_mean'] * (500 / x[1]['width_mean'])
        )[0]
        
        print(f"\n✓ Identified Cluster {periapical_cluster_id} as PERIAPICAL")
        return periapical_cluster_id
    
    def organize_images(self, image_paths, labels, periapical_cluster_id, confidence_threshold=0.8):
        periapical_count = 0
        non_periapical_count = 0
        uncertain_count = 0
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                img_name = img_path.name
                
                if label == periapical_cluster_id:
                    dest = self.periapical_dir / img_name
                    shutil.copy2(img_path, dest)
                    periapical_count += 1
                else:
                    dest = self.non_periapical_dir / img_name
                    shutil.copy2(img_path, dest)
                    non_periapical_count += 1
                    
            except Exception as e:
                continue
        if uncertain_count > 0:
            pass
    
    def visualize_clusters(self, features, labels, periapical_cluster_id, save_path=None):        
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
    
        plt.figure(figsize=(12, 8))
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_name = "Periapical" if cluster_id == periapical_cluster_id else "Non-Periapical"
            color = 'blue' if cluster_id == periapical_cluster_id else 'red'
            
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                label=f"{cluster_name} (n={sum(mask)})",
                alpha=0.6,
                s=50,
                c=color
            )
        
        plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
        plt.title('X-ray Image Clustering: Periapical vs Non-Periapical', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
    
    def run(self, visualize=True):
        print("\n" + "="*80)
        print("AUTOMATIC X-RAY IMAGE CLUSTERING")
        print("="*80)
        print(f"Source folder: {self.source_folder}")
        print(f"Target folder: {self.target_folder}")
        print("="*80)
        
        image_paths = self.find_all_images()
        
        if len(image_paths) == 0:
            return None
        features, valid_paths = self.extract_features(image_paths)
        print("features",features)
        labels, kmeans, pca = self.cluster_images(features, n_clusters=2)
        periapical_cluster_id = self.identify_periapical_cluster(features, labels, valid_paths)
        self.organize_images(valid_paths, labels, periapical_cluster_id)
        
        if visualize:
            viz_path =str("results/clustering_visualization.png")
            self.visualize_clusters(features, labels, periapical_cluster_id, viz_path)
        stats = {
            'total_images': len(valid_paths),
            'periapical_count': sum(labels == periapical_cluster_id),
            'non_periapical_count': sum(labels != periapical_cluster_id),
            'periapical_dir': str(self.periapical_dir),
            'non_periapical_dir': str(self.non_periapical_dir)
        }
        
        print("\n" + "="*80)
        print("CLUSTERING COMPLETE!")
        print("="*80)
        print(f"Total images processed: {stats['total_images']}")
        print(f"Periapical:     {stats['periapical_count']} images")
        print(f"Non-Periapical: {stats['non_periapical_count']} images")
        print("\nOrganized folders:")
        print(f"{stats['periapical_dir']}")
        print(f"{stats['non_periapical_dir']}")
        print("="*80)
        
        return stats


def manual_review_interface(periapical_dir, non_periapical_dir, num_samples=10):
    periapical_images = list(Path(periapical_dir).glob('*.jpg')) + \
                       list(Path(periapical_dir).glob('*.png'))
    non_periapical_images = list(Path(non_periapical_dir).glob('*.jpg')) + \
                           list(Path(non_periapical_dir).glob('*.png'))
    
    periapical_samples = np.random.choice(periapical_images, 
                                         min(num_samples, len(periapical_images)), 
                                         replace=False)
    non_periapical_samples = np.random.choice(non_periapical_images, 
                                             min(num_samples, len(non_periapical_images)), 
                                             replace=False)
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(20, 8))
    fig.suptitle('PERIAPICAL Samples (Review for Accuracy)', fontsize=16, fontweight='bold')
    
    for idx, (ax, img_path) in enumerate(zip(axes.ravel(), periapical_samples)):
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.set_title(img_path.name, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path('results')/ 'periapical_samples.png', dpi=150)
    plt.show()
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(20, 8))
    fig.suptitle('NON-PERIAPICAL Samples (Review for Accuracy)', fontsize=16, fontweight='bold')
    
    for idx, (ax, img_path) in enumerate(zip(axes.ravel(), non_periapical_samples)):
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.set_title(img_path.name, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path('results') / 'non_periapical_samples.png', dpi=150)
    plt.show()
    
    print("\n✓ Review images saved!")
    print(f"  Periapical samples: {Path('results') / 'periapical_samples.png'}")
    print(f"  Non-Periapical samples: {Path('results') / 'non_periapical_samples.png'}")


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Automatic X-ray Image Clustering')
#     parser.add_argument('--source', type=str, required=True,
#                        help='Source folder containing all X-ray images')
#     parser.add_argument('--target', type=str, default=None,
#                        help='Target folder to save organized images (default: data/raw)')
#     parser.add_argument('--no-viz', action='store_true',
#                        help='Skip visualization')
#     parser.add_argument('--review', action='store_true',
#                        help='Show manual review interface after clustering')
    
#     args = parser.parse_args()
    
#     clusterer = AutomaticImageClusterer(args.source, args.target)
#     stats = clusterer.run(visualize=not args.no_viz)
    
#     if stats and args.review:
#         manual_review_interface(
#             stats['periapical_dir'],
#             stats['non_periapical_dir'],
#             num_samples=10
#         )


# if __name__ == "__main__":
#     print("Automatic X-ray Clustering")
#     print("="*80)
    
#     # Ask user for source folder
#     source_folder = input("Enter path to folder with all X-ray images: ").strip()
    
#     if not Path(source_folder).exists():
#         print(f"❌ Folder not found: {source_folder}")
#         exit(1)
    
#     # Run clustering
#     clusterer = AutomaticImageClusterer(source_folder)
#     stats = clusterer.run(visualize=True)
    
#     if stats:
#         # Ask for manual review
#         review = input("\nWould you like to review sample images? (y/n): ").strip().lower()
#         if review == 'y':
#             manual_review_interface(
#                 stats['periapical_dir'],
#                 stats['non_periapical_dir']
#             )