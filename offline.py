from PIL import Image
from pathlib import Path
import numpy as np
from featureExtractor import FeatureExtractor

if __name__ == "__main__":
    extractor = FeatureExtractor()
    for img_path in sorted(Path("./static/img").glob("*.jpeg")):

        print("imagePath")

        #feature extraction
        feature = extractor.extract(img=Image.open(img_path))
        print(type(feature), feature.shape)

        feature_path = Path('./static/feature') / (img_path.stem + ".npy")
        print(feature_path)
        np.save(feature_path, feature)
