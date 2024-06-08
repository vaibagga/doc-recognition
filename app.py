import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from json import JSONEncoder

from src.models import UNet

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Initialize the FastAPI application
app = FastAPI()

# Load your pre-trained PyTorch model
model = UNet()
model.load_state_dict(torch.load('model/UNET_Adam.pth'))

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust the target size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image):
    """Preprocess the image to the required input shape for the model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image prediction requests."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)
        ##print(processed_image)
        with torch.no_grad():
            outputs = model(processed_image)

        encoded_numpy = json.dumps(outputs.detach().numpy()[0,0,:,:], cls=NumpyArrayEncoder)


        response = {
            "predicted_class": encoded_numpy,
            "code": 200
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_on_video")
async def predict(file: UploadFile = File(...)):
    """Handle image prediction requests."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)
        ##print(processed_image)
        with torch.no_grad():
            outputs = model(processed_image)

        encoded_numpy = json.dumps(outputs.detach().numpy()[0,0,:,:], cls=NumpyArrayEncoder)


        response = {
            "predicted_class": encoded_numpy,
            "code": 200
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
