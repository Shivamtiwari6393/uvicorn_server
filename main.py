# main.py
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model 
import cv2


app = FastAPI()


model = load_model("models/custom_asl_model_v2_Z.keras")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello/")
def say_hello():
    return {"message": f"Hello"}


@app.post("/upload/")
async def upload_image(
    image: UploadFile = File(...)
):
    label = "A"
    UPLOAD_DIR = os.path.join("uploaded_images_v2/test", label)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{label}_{image.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return JSONResponse(
        content={"message": "Image uploaded successfully", "filename": filename},
        status_code=200
    )


@app.post("/predict/")
async def predict_image(
    image: UploadFile = File(...)
):

    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # img = cv2.imread('C:/Users/91639/Pictures/python_model/augmented_dataset/B/B_0_1018.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (128,128))
    img = img.reshape(-1, 128,128,3)
    # cv2.imshow(img)

    # Predict
    pred = model.predict(img)
    predicted_class = pred.argmax()
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    out = classes[predicted_class]
    accu = pred[0][predicted_class]

    print(out, accu)

    return JSONResponse(
        content={"message": f"Prediction successful acc {accu*100}", "filename": out},
        status_code=200
    )
