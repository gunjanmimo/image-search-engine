import numpy as np
from PIL import Image
from featureExtractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)
extractor = FeatureExtractor()
# read image feature
features = []
image_path = []
for feature_path in Path("./static/feature/").glob("*.npy"):
    features.append(np.load(feature_path))
    image_path.append(Path("./static/img/") / (feature_path.stem + ".jpeg"))

features = np.array(features)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # save image
        img = Image.open(file.stream)
        upload_img_path = "static/uploaded/"+\
            datetime.now().isoformat().replace(":",".")+"_"+file.filename

        img.save(upload_img_path)

        # search
        query = extractor.extract(img)
        dists = np.linalg.norm(features - query, axis=1)

        ids = np.argsort(dists)[:30]
        scores = [(dists[id], image_path[id]) for id in ids]
        print(scores)
        return render_template('index.html',
                               query_path=upload_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()