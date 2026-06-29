from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from utils.preprocessing import preprocess_image
from services.model_loader import predict

app = FastAPI(
    title="Fish Freshness API",
    description="API Prediksi Kesegaran Ikan Menggunakan MobileNetV3-Large + SVM",
    version="1.0.0"
)

# Mengizinkan akses dari Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Fish Freshness API"
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:

        # Membaca gambar
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocessing
        image_tensor = preprocess_image(image)

        # Prediksi
        result = predict(image_tensor)

        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }