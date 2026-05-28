import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn import svm
import joblib

# Load AlexNet
alexnet = models.alexnet(pretrained=True)
alexnet.classifier = torch.nn.Sequential(*list(alexnet.classifier.children())[:-1])
alexnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = alexnet(img)
    return features.numpy().flatten()

X = []
y = []

# dataset_path = "dataset\MICC-F220"
dataset_path = "MICC-F220"

# dataset/
#   real/
#   forged/

for label, folder in enumerate(["Au", "Tu"]):
    folder_path = os.path.join(dataset_path, folder)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        feat = extract_features(img)
        X.append(feat)
        y.append(label)

# Train SVM
model_1 = svm.SVC(kernel='linear', probability=True)
model_1.fit(X, y)

# Save model
joblib.dump(model_1, "model/1_svm_model.pkl")

print("✅ SVM model trained and saved!")