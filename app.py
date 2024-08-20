from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
from deepface import DeepFace
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

app = FastAPI()

qdrant_client = QdrantClient(
    url="https://f575a6c0-4880-4ba1-bac4-ce3febd5e83a.us-east4-0.gcp.cloud.qdrant.io",
    api_key="P9qpgcryuCBF8c78hYU4QwImfJIkZimHXZSIBqfax6W3rGi-LNwN4g",
)

@app.post("/search-face")
async def upload_image(image: UploadFile = File(...),):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    image_bytes = await image.read()
    image_np = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    girl_em = DeepFace.represent(img, model_name = 'Facenet')


  

    result = qdrant_client.search(
            collection_name="face_data",
            query_vector=girl_em[0]['embedding'],
            limit=1,
        )


    return JSONResponse(content={
        "adhaar": result[0].id,
        
    })


@app.post("/upload-face")
async def upload_image(image: UploadFile = File(...),some_number: int = Form(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    image_bytes = await image.read()
    image_np = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    girl_em = DeepFace.represent(img, model_name = 'Facenet')



    try:
        qdrant_client.get_collection("face_data")
    except:
        print("collection not found")

    qdrant_client.upsert(
        collection_name="face_data",
        points=[
            {
                "id": some_number,
                "vector": girl_em[0]['embedding'],

            }
        ]
    )



    return JSONResponse(content={
        "message": "image upload done",
        
    })


@app.get("/")
def read_root():
    return {"message": "Try the endpoints /search-face or /upload-face"}

