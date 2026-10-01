import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.applications.resnet50 import preprocess_input

# Dictionary container to hold our machine learning model globally in memory
ml_models = {}

# 1. LIFESPAN MANAGEMENT: Loads the heavy model into memory EXACTLY ONCE on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🧠 Loading heavy ResNet50 model architecture into system RAM...")
    # This takes a few seconds on boot, but saves massive latency on uploads!
    ml_models["shelf_model"] = tf.keras.models.load_model('shelf_monitor_model.keras')
    print("🚀 Model loaded successfully. Control center is active and optimized!")
    yield
    # Clean up memory when the server shuts down
    ml_models.clear()

# Initialize FastAPI with the lifespan context manager
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

CLASS_LABELS = ['Fully Stocked', 'Low Stock', 'Out of Stock']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict_inventory(request: Request, file: UploadFile = File(...)):
    # Read image data stream instantly
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Generate Base64 preview
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Preprocess dimensions rapidly 
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 2. INSTANT INFERENCE: Grab the pre-warmed model directly from RAM cache
    model = ml_models["shelf_model"]
    prediction = model.predict(img_array)
    
    class_idx = np.argmax(prediction)
    status = CLASS_LABELS[class_idx]
    confidence = float(prediction[0][class_idx]) * 100

    # Business evaluation logic
    if status == 'Out of Stock':
        alert_action = "🔴 CRITICAL ALERT: Immediate Restock Required!"
        badge_class = "danger"
    elif status == 'Low Stock':
        alert_action = "🟡 WARNING: Schedule inventory replenishment soon."
        badge_class = "warning"
    else:
        alert_action = "🟢 OPTIMAL: Shelf capacity is running stable."
        badge_class = "success"

    result = {
        "filename": file.filename,
        "status": status,
        "confidence": f"{confidence:.2f}%",
        "action": alert_action,
        "badge": badge_class,
        "image_data": f"data:image/jpeg;base64,{img_str}"
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result})

if __name__ == "__main__":
    import uvicorn
    # Turned off reload loop to protect memory stability on Mac architectures
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)
