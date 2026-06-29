import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SVM_PATH = os.path.join(MODEL_DIR, "svm_mbv3_large.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_mbv3_large.pkl")
MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenetv3_large.pth")

LABEL_NAMES = [
    "Highly Fresh",
    "Fresh",
    "Not Fresh"
]


def build_extractor():

    model = models.mobilenet_v3_large(weights=None)

    extractor = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1)
    )

    # Jika ada file hasil save state_dict
    if os.path.exists(MOBILENET_PATH):
        extractor.load_state_dict(
            torch.load(
                MOBILENET_PATH,
                map_location=DEVICE
            )
        )
        print("MobileNetV3 berhasil dimuat.")
    else:
        print("mobilenetv3_large.pth tidak ditemukan.")
        print("Menggunakan pretrained ImageNet.")

    extractor.to(DEVICE)
    extractor.eval()

    return extractor


# Load MobileNet
extractor = build_extractor()


# Load Scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Scaler berhasil dimuat.")


# Load SVM
with open(SVM_PATH, "rb") as f:
    svm = pickle.load(f)

print("SVM berhasil dimuat.")


def extract_feature(image_tensor):
    """
    Mengubah tensor gambar menjadi feature 960 dimensi.

    Parameters
    ----------
    image_tensor : Tensor
        Shape (1,3,224,224)

    Returns
    -------
    numpy.ndarray
        Shape (1,960)
    """

    with torch.no_grad():

        feature = extractor(
            image_tensor.to(DEVICE)
        )

        feature = feature.squeeze(-1).squeeze(-1)

        feature = feature.cpu().numpy()

    return feature


def predict(image_tensor):
    """
    Prediksi kelas kesegaran ikan.

    Returns
    -------
    dict
    """

    feature = extract_feature(image_tensor)

    feature = scaler.transform(feature)

    prediction = svm.predict(feature)[0]

    probability = svm.predict_proba(feature)[0]

    confidence = float(probability.max() * 100)

    return {
        "prediction": LABEL_NAMES[prediction],
        "confidence": round(confidence, 2),
        "probabilities": {
            LABEL_NAMES[i]: round(float(probability[i]) * 100, 2)
            for i in range(len(LABEL_NAMES))
        }
    }