# from PIL import Image  # Import Pillow for image processing
# import numpy as np
# from flask import jsonify
# import json
# import time
# import tensorflow as tf

# # Get absolute paths
# from pathlib import Path
# import os


# class_labels_disease = [
#     "bacterial_leaf_blight",
#     "bacterial_leaf_streak",
#     "bacterial_panicle_blight",
#     "blast",
#     "brown_spot",
#     "dead_heart",
#     "downy_mildew",
#     "hispa",
#     "normal",
#     "tungro",
# ]

# class_labels_varieties = [
#     "ADT45",
#     "AndraPonni",
#     "AtchayaPonni",
#     "IR20",
#     "KarnatakaPonni",
#     "Onthanel",
#     "Ponni",
#     "RR",
#     "Surya",
#     "Zonal",
# ]


# BASE_DIR = Path(__file__).resolve().parent

# # UI folder is sibling to Server
# UI_DIR = os.path.join(BASE_DIR.parent, "UI")

# # Uploads and models are inside Server
# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# PIC_DIR = os.path.join(UPLOAD_DIR, "pic")
# DATABASE_DIR = os.path.join(BASE_DIR.parent, "database")


# def loader(model_name, model_path):
#     print(f"Loading {model_name} model...")
#     start_time = time.time()
#     model = tf.keras.models.load_model(model_path)
#     print(f"{model_name} model loaded in {time.time() - start_time:.2f} seconds.")
#     return model


# def load_all():
#     global task1_model_vgg, task1_model_mobile_net, task1_model_cnn
#     global task2_resnet50_model, task2_efficient_net_model, task2_model_cnn
#     global task3_nasnet_model, task3_model_cnn
#     # Load models
#     task1_model_vgg = loader(
#         "VGG", "../database/models/Task1/task1_vgg_model.h5")
#     task1_model_mobile_net = loader(
#         "MobileNet", "../database/models/Task1/task1_mobile_net_model.h5"
#     )
#     task1_model_cnn = loader(
#         "CNN", "../database/models/Task1/task1_cnn_model.h5")

#     task2_resnet50_model = loader(
#         "Resnet50", "../database/models/Task2/task2_resnet50_model.h5"
#     )
#     task2_efficient_net_model = loader(
#         "EfficientNet", "../database/models/Task2/task2_efficient_net_model.h5"
#     )
#     task2_model_cnn = loader(
#         "CNN", "../database/models/Task2/task2_cnn_model.h5")

#     task3_nasnet_model = loader(
#         "Nasnet", "../database/models/Task3/task3_nasnet_model.h5"
#     )
#     task3_model_cnn = loader(
#         "CNN", "../database/models/Task3/task3_cnn_model.h5")


# # normalize the image


# def preprocess_image(img_path):
#     img = Image.open(img_path)
#     img = img.resize((224, 224))  # Resize to the target size
#     img_array = np.array(img) / 255.0  # Normalize to [0, 1]
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array


# def classify(file, file_path, mode):
#     # Predict with all models
#     img_array = preprocess_image(file_path)

#     try:
#         if mode == "disease":
#             vgg_prediction = task1_model_vgg.predict(img_array)
#             mobile_net_prediction = task1_model_mobile_net.predict(img_array)
#             cnn_prediction = task1_model_cnn.predict(img_array)

#             vgg_index = int(np.argmax(vgg_prediction, axis=1)[0])
#             vgg_confidence = float(np.max(vgg_prediction, axis=1)[0])
#             vgg_label = class_labels_disease[vgg_index]

#             mobile_index = int(np.argmax(mobile_net_prediction, axis=1)[0])
#             mobile_confidence = float(np.max(mobile_net_prediction, axis=1)[0])
#             mobile_label = class_labels_disease[mobile_index]

#             cnn_index = int(np.argmax(cnn_prediction, axis=1)[0])
#             cnn_confidence = float(np.max(cnn_prediction, axis=1)[0])
#             cnn_label = class_labels_disease[cnn_index]

#             predictions = {
#                 "vgg": {
#                     "label": vgg_label,
#                     "confidence": f"{vgg_confidence * 100:.2f}%",
#                 },
#                 "mobile_net": {
#                     "label": mobile_label,
#                     "confidence": f"{mobile_confidence * 100:.2f}%",
#                 },
#                 "cnn": {
#                     "label": cnn_label,
#                     "confidence": f"{cnn_confidence * 100:.2f}%",
#                 },
#             }

#         if mode == "variety":
#             resnet50_prediction = task2_resnet50_model.predict(img_array)
#             efficient_net_prediction = task2_efficient_net_model.predict(
#                 img_array)
#             cnn_prediction = task2_model_cnn.predict(img_array)

#             resnet50_index = int(np.argmax(resnet50_prediction, axis=1)[0])
#             resnet50_confidence = float(np.max(resnet50_prediction, axis=1)[0])
#             resnet50_label = class_labels_varieties[resnet50_index]

#             efficient_net_index = int(
#                 np.argmax(efficient_net_prediction, axis=1)[0])
#             efficient_net_confidence = float(
#                 np.max(efficient_net_prediction, axis=1)[0]
#             )
#             efficient_net_label = class_labels_varieties[efficient_net_index]

#             cnn_index = int(np.argmax(cnn_prediction, axis=1)[0])
#             cnn_confidence = float(np.max(cnn_prediction, axis=1)[0])
#             cnn_label = class_labels_varieties[cnn_index]

#             predictions = {
#                 "resnet50": {
#                     "label": resnet50_label,
#                     "confidence": f"{resnet50_confidence * 100:.2f}%",
#                 },
#                 "efficient_net": {
#                     "label": efficient_net_label,
#                     "confidence": f"{efficient_net_confidence * 100:.2f}%",
#                 },
#                 "cnn": {
#                     "label": cnn_label,
#                     "confidence": f"{cnn_confidence * 100:.2f}%",
#                 },
#             }

#         if mode == "age":
#             nasnet_prediction = task3_nasnet_model.predict(img_array)
#             cnn_prediction = task3_model_cnn.predict(img_array)

#             nasnet_index = nasnet_prediction[0][0][0]

#             cnn_index = cnn_prediction[0][0]

#             predictions = {
#                 "nasnet": {
#                     "label": f"{nasnet_index * 80:.2f}",
#                 },
#                 "cnn": {
#                     "label": f"{cnn_index * 80:.2f}",
#                 },
#             }
#             # Save to predictions.json
#         prediction_file_path = os.path.join(DATABASE_DIR, "predictions.json")
#         data = {file.filename: predictions}

#         with open(prediction_file_path, "w") as f:
#             json.dump(data, f, indent=4)

#         return jsonify(
#             {
#                 "predictions": predictions,
#                 "message": "Image classified and predictions saved successfully",
#             }
#         )
#     except Exception as e:
#         print(f"Error during classification: {e}")
#         return jsonify({"success": False, "error": f"Prediction failed"}), 500

import numpy as np
from sklearn.linear_model import LogisticRegression

# train once when module loads
def load_model():
    X = np.array([
        [0], [1], [2], [3]
    ])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)
    return model

model = load_model()

def predict(value):
    value = np.array([[value]])
    pred = model.predict(value)
    return int(pred[0])