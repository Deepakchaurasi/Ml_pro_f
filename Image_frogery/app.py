import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
from predict import predict
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
UPLOAD_FOLDER2= "uploads_old"
REAL_FOLDER = "real_images"
REAL_FOLDER2 = "real_images_old"
# model = tf.keras.models.load_model("model/new_model_train.keras") creat your module .
IMG_SIZE = 224
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["UPLOAD_FOLDER2"] = UPLOAD_FOLDER2
app.config["REAL_FOLDER"] = REAL_FOLDER
app.config["REAL_FOLDER2"] = REAL_FOLDER2
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER2, exist_ok=True)
os.makedirs(REAL_FOLDER, exist_ok=True)
os.makedirs(REAL_FOLDER2, exist_ok=True)

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    return img
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/index_old")
def new_version():
    return render_template("index_old.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/loin")
def login():
    return render_template("loin.html")
@app.route("/res")
def register():
    return render_template("res.html")
@app.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if(not file.filename.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))):
        return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."})
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    if file.filename.lower().endswith(".tif"):
        img = Image.open(filepath)
        clean_name = os.path.splitext(file.filename)[0]
        new_filename = clean_name + ".png"
        new_filepath = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
        img.convert("RGB").save(new_filepath, "PNG")  
        os.remove(filepath)  
        filepath = new_filepath
        # filepath = new_filepath  
    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # img = preprocess_image(filepath)
    # prediction = model.predict(img_array)
    prediction = prediction[0][0] 
    print(f"Prediction: {prediction}")

    result = "Forged" if prediction > 0.5 else "Real"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    if result == "Real":
        output = "REAL"
        # filepath1 = os.path.join(app.config["REAL_FOLDER"], file.filename)
        # if(not os.path.exists(filepath1)):
        #    shutil.copyfile(filepath, filepath1)
    else:
        output = "FORGED"

    return jsonify({"result": output, "confidence": confidence})
@app.route("/predict_old", methods=["POST"])
def predect_old():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if(not file.filename.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))):
        return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."})
    filepath = os.path.join(app.config["UPLOAD_FOLDER2"], file.filename)
    file.save(filepath)

    image = Image.open(filepath)
    result, confidence = predict(image)

    print(f"Prediction: {confidence}")
    
    if result == 0:
        output = "REAL"
        # filepath1 = os.path.join(app.config["REAL_FOLDER2"], file.filename)
        # if(not os.path.exists(filepath1)):
        #    shutil.copyfile(filepath, filepath1)
    else:
        output = "FORGED"

    return jsonify({"result": output, "confidence": confidence})
if __name__ == "__main__":
    app.run(debug=True)