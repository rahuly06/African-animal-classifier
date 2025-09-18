from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io
from .CNNModel_class_transfer import CNNModel  # import your model class

app = FastAPI()

# Load model and weights


model = CNNModel(num_classes=5)  # define your model class
state_dict = torch.load("app/resnet18_african_animals.pth", map_location="cpu")
model.resnet.load_state_dict(state_dict)
model.eval()

# Define transforms (same as training)
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.get("/")
async def root():
    return {"message": "Welcome to the African Animal Classifier API!"}


# List of class names (update this to match your dataset/class order)
class_names = [
    'african Rhinoceros',
    'african buffalo',
    'african forest elephants',
    'african lion',
    'african zebra'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    class_idx = int(predicted.item())
    class_name = class_names[class_idx]
    return {"class_index": class_idx, "class_name": class_name}