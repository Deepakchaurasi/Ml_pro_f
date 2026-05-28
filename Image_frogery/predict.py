import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
import numpy as np

# Load AlexNet
alexnet = models.alexnet(pretrained=True)
alexnet.classifier = torch.nn.Sequential(*list(alexnet.classifier.children())[:-1])
alexnet.eval()

# Load SVM
# svm_model = joblib.load("model/1_svm_model.pkl")

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

def extract_features(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = alexnet(image)   
    return features.numpy().flatten()

def predict(image):
    features = extract_features(image)
    
    # Prediction
    prediction = svm_model.predict([features])[0]
    
    # Confidence (probability)
    if hasattr(svm_model, "predict_proba"):
        confidence = np.max(svm_model.predict_proba([features]))
    else:
        # fallback if probability not enabled
        confidence = None
    
    return prediction, confidence