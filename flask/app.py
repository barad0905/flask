from flask import Flask, request, jsonify
from utils.detection import detect_road, detect_trees
from utils.georeference import get_georeference_data, pixel_to_geo
from utils.measurements import calculate_metrics, extend_road, count_trees
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

road_model = tf.keras.models.load_model('model/road_unet_resnet.keras')
tree_model = tf.keras.models.load_model('model/tree_unet_resnet_finetune.keras')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    crs, transform, bounds = get_georeference_data(file_path)
    road_mask = detect_road(file_path, road_model)
    road_polygons = pixel_to_geo(road_mask, transform)

    tree_mask = detect_trees(file_path, tree_model)
    tree_polygons = pixel_to_geo(tree_mask, transform)

    metrics = [calculate_metrics(polygon) for polygon in road_polygons]

    return jsonify({
        'crs': str(crs),
        'road_polygons': [p.__geo_interface__ for p in road_polygons],
        'tree_polygons': [p.__geo_interface__ for p in tree_polygons],
        'metrics': metrics
    })

@app.route('/extend', methods=['POST'])
def extend_road_api():
    data = request.json
    polygons = [Polygon(p['coordinates'][0]) for p in data['road_polygons']]
    extension_width = float(data['extension_width'])
    extended_polygons = [extend_road(p, extension_width) for p in polygons]
    tree_polygons = [Polygon(p['coordinates'][0]) for p in data['tree_polygons']]
    tree_count = sum(count_trees(tree_polygons, ep) for ep in extended_polygons)

    return jsonify({
        'extended_polygons': [p.__geo_interface__ for p in extended_polygons],
        'tree_count': tree_count
    })

if __name__ == '__main__':
    app.run(debug=True)
