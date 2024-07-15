import os
import io
import zipfile
from flask import Flask, request, render_template, jsonify, send_file, abort, redirect, url_for
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs
from detect import run
import torch

# MongoDB connection string
uri = "mongodb+srv://kushiluv:kushiluv25@cluster0.pety1ki.mongodb.net/"
client = MongoClient(uri)
db = client['ImageDatabase']
fs = gridfs.GridFS(db)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Clear old data
        db.fs.chunks.drop()
        db.fs.files.drop()
        db.CategorizedImages.drop()
        
        files = request.files.getlist('file')
        file_ids = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_id = fs.put(file.read(), filename=filename, content_type=file.content_type)
                file_ids.append(str(file_id))
        
        # Create a single string from the list of file IDs
        file_ids_str = ','.join(file_ids)
        
        # Print file IDs for debugging
        print("File IDs:", file_ids_str)
        
        # Call the run function from detect.py
        run(
            weights='runs/train/wii_28_072/weights/best.pt',
            data='data/wii_aite_2022_testing.yaml',
            imgsz=(640, 640),
            conf_thres=0.001,
            iou_thres=0.6,
            max_det=1000,
            device='0' if torch.cuda.is_available() else 'cpu',
            view_img=False,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            nosave=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            visualize=False,
            project='runs/detect',
            name='yolo_test_24_08_site0001',
            exist_ok=False,
            line_thickness=3,
            hide_labels=False,
            hide_conf=False,
            half=False,
            dnn=False,
            mongodb_uri=uri,
            file_ids=file_ids_str
        )
        
        return redirect(url_for('results'))

    categories = db['CategorizedImages'].distinct('category')
    categorized_images = {}
    for category in categories:
        images = db['CategorizedImages'].find({'category': category})
        categorized_images[category] = images
    return render_template('results.html', categories=categories, categorized_images=categorized_images)

@app.route('/results', methods=['GET'])
def results():
    categories = db['CategorizedImages'].distinct('category')
    categorized_images = {}
    for category in categories:
        images = db['CategorizedImages'].find({'category': category})
        categorized_images[category] = images
    return render_template('results.html', categories=categories, categorized_images=categorized_images)

@app.route('/download', methods=['GET'])
def download_all():
    categories = db['CategorizedImages'].distinct('category')
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for category in categories:
            images = db['CategorizedImages'].find({'category': category})
            for image in images:
                file_id = image['file_id']
                image_doc = fs.get(ObjectId(file_id))
                image_name = os.path.join(category, image_doc.filename)
                zf.writestr(image_name, image_doc.read())
    
    zip_buffer.seek(0)
    
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='categorized_images.zip')

@app.route('/image/<id>', methods=['GET'])
def display_image(id):
    try:
        image_doc = fs.get(ObjectId(id))
        return send_file(io.BytesIO(image_doc.read()), mimetype=image_doc.content_type)
    except Exception as e:
        abort(404, description=f"Image not found: {e}")

if __name__ == '__main__':
    app.run(debug=True)
