from PIL import Image
import torchvision.transforms as transforms

# Ukuran input MobileNetV3
CNN_SIZE = (224, 224)

# Transform sama persis seperti saat training
cnn_transform = transforms.Compose([
    transforms.Resize(CNN_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image):
    """
    Mengubah gambar PIL menjadi tensor yang siap diproses MobileNetV3.

    Parameter
    ---------
    image : PIL.Image

    Return
    ------
    Tensor dengan shape (1, 3, 224, 224)
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = cnn_transform(image)

    # Tambahkan batch dimension
    image = image.unsqueeze(0)

    return image