from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_path = "weather_classifier.pth"
num_classes = 11
class_names = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

##transforming stuff YAY

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):  
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_idx = torch.max(outputs, 1)

    pred_label = class_names[pred_idx.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred_idx.item()].item()

    return {"prediction": pred_label, "confidence": round(confidence, 4)}