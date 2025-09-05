from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io
from CNNModel_class import CNNModel  # import your model class

app = FastAPI()

# Load model and weights

model = CNNModel(input_chanels=3)  # define your model class
model.load_state_dict(torch.load("animal_classifier.pth", map_location="cpu"))
model.eval()

# Define transforms (same as training)
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.get("/")
async def root():
    return {"message": "Welcome to the African Animal Classifier API!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return {"class": int(predicted.item())}