import logging
import traceback  
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from io import BytesIO
from pydantic import BaseModel
from fastapi.responses import FileResponse
import torch
from PIL import Image
import nibabel as nib
import cv2
import json
import numpy as np
import base64
import matplotlib.pyplot as plt
from rayemb.models import RayEmbSubspaceInference  # Adjust the import based on your model
from rayemb.utils import cosine_similarity, find_batch_peak_coordinates_maxval  # Adjust the import based on your similarity function
from server.constants import CHECKPOINT_PATH
import aioredis
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
redis = None

# Add this near the top of the file with other constants
UPLOAD_DIR = "../tmp"  # You can change this path to your preferred directory

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

class Location(BaseModel):
    x: float
    y: float
    z: float

# Function to convert NumPy array to image and send as response
def numpy_to_image_response(np_array):
    image = Image.fromarray(np_array.astype('uint8'))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def numpy_to_image_response_with_metadata(np_array, metadata):
    # Convert NumPy array to image
    image = Image.fromarray(np_array.astype('uint8'))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    
    # Encode image to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Create a JSON response with image and metadata
    response_data = {
        "image": image_base64,
        "metadata": metadata
    }
    
    return response_data

# Function to convert NumPy array to jet colormap image and send as response
def numpy_to_colormap_image_response(np_array, pred_proj_point=None, gt_proj_point=None):
    # Normalize the array to the range [0, 1]
    norm_array = (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))
    
    # Apply the jet colormap
    colormap = plt.get_cmap('jet')
    colored_array = colormap(norm_array)
    
    # Convert to 8-bit unsigned integer
    colored_array = (colored_array[:, :, :3] * 255).astype(np.uint8)
    colored_array = cv2.resize(colored_array, (512, 512))
    if pred_proj_point is not None:
        # draw the predicted point on the heatmap using cv2
        # Draw a white border circle
        cv2.circle(colored_array, (int(pred_proj_point[0]*512/np_array.shape[0]), int(pred_proj_point[1]*512/np_array.shape[1])), 8, (255, 255, 255), -1)
        # Draw a smaller red inner circle
        cv2.circle(colored_array, (int(pred_proj_point[0]*512/np_array.shape[0]), int(pred_proj_point[1]*512/np_array.shape[1])), 5, (0, 0, 255), -1)
    if gt_proj_point is not None:
        # Draw a white border circle
        cv2.circle(colored_array, (int(gt_proj_point[0]), int(gt_proj_point[1])), 8, (255, 255, 255), -1)
        # Draw a smaller red inner circle
        cv2.circle(colored_array, (int(gt_proj_point[0]), int(gt_proj_point[1])), 5, (255, 0, 0), -1)
    # Convert to PIL image
    image = Image.fromarray(colored_array)
    # Resize to 512 x 512
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Load the model during startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global redis
    global spacing
    global ct_transform
    # global query_projection_matrix
    model = RayEmbSubspaceInference.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        map_location=device,
        similarity_fn=cosine_similarity,
        emb_dim=32,
        image_size=224
    )
    model.eval()
    logger.info("Model loaded successfully")
    # Connect to Redis
    redis = aioredis.from_url("redis://localhost", decode_responses=True)
    ct_path = "../ui/public/assets/test.nii.gz" # This is the downsized dataset6_CLINIC_0001_data
    xray_path = "../data/ctpelvic1k_synthetic/dataset6_CLINIC_0001_data/images/0010.png"
    camera_path = "../data/ctpelvic1k_synthetic/dataset6_CLINIC_0001_data/cameras/0010.json"
    template_path = "../data/ctpelvic1k_templates/dataset6_CLINIC_0001_data"
    # For now load a fixed ct and x-ray
    await redis.set("ct", ct_path)
    await redis.set("xray", xray_path)
    await redis.set("template", template_path)
    await redis.set("camera", camera_path)
    ct_img = nib.load(ct_path)
    ct_transform = np.array(ct_img.affine)
    ct_transform = np.linalg.inv(ct_transform)
    spacing = np.array(ct_img.header.get_zooms())
    image_array = np.array(Image.open(xray_path))
    image_array = np.repeat(image_array[..., np.newaxis], 3, axis=-1)
    # query_projection_matrix = np.array(json.load(open(camera_path))["proj"])[0]
    model.set_templates(template_path)
    model.set_image(image_array)
    model.calc_features()
    yield
    # Clean up resources
    del model
    torch.cuda.empty_cache()
    await redis.close()

app = FastAPI(lifespan=lifespan)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origins you want to allow
    allow_credentials=False,
    allow_methods=["*"],  # Adjust this to the specific methods you want to allow
    allow_headers=["*"],  # Adjust this to the specific headers you want to allow
)

# @app.post("/image/ct")
# async def set_ct(ct_path: str = "../data/CTPelvic1K/dataset6_volume/dataset6_CLINIC_0001_data.nii.gz",
#                  template_path: str = "../data/ctpelvic1k_templates_v2/dataset6_CLINIC_0001_data"):
#     global redis, model, ct_transform, spacing
#     model, redis = check_model_and_redis(model, redis)
#     try:
#         await redis.set("ct", ct_path)
#         await redis.set("template", template_path)
#         model.set_templates(template_path)
#         ct_img = nib.load(ct_path)
#         ct_transform = np.array(ct_img.affine)
#         ct_transform = np.linalg.inv(ct_transform)
#         spacing = np.array(ct_img.header.get_zooms())
#         return {"message": "CT Templates set"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/ct")
async def get_ct():
    global redis, model
    model, redis = check_model_and_redis(model, redis)
    try:
        return FileResponse(await redis.get("ct"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/xray")
async def get_xray():
    global redis, model
    model, redis = check_model_and_redis(model, redis)
    try:
        return FileResponse(await redis.get("xray"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image/xray")
async def set_xray(file: UploadFile = File(...)):
    global model, redis
    model, redis = check_model_and_redis(model, redis)
    try:
        # Create a file path in the specified directory
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Read and save the uploaded file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Store the file path in Redis
        await redis.set("xray", file_path)
        
        # Process the image for the model
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # If grayscale, convert to RGB
            image_array = np.repeat(image_array[..., np.newaxis], 3, axis=-1)
        
        # Update the model with new image
        model.set_image(image_array)
        model.calc_features()
        
        return {"success": True, "filename": file.filename}
    except Exception as e:
        # Clean up file in case of error
        if 'file_path' in locals():
            os.unlink(file_path)
        tb_info = traceback.format_exc()
        logger.error(tb_info)
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/image/xray")
# async def set_xray(image_path: str = "../data/ctpelvic1k_synthetic_v2/dataset6_CLINIC_0001_data/images/0010.png"):
#     global model, redis
#     model, redis = check_model_and_redis(model, redis)
#     try:
#         # Process the uploaded file do this later
#         # contents = await file.read()
#         # image = Image.open(BytesIO(contents))
#         # image_array = np.array(image)
#         await redis.set("xray", image_path)
#         image_array = np.array(Image.open(image_path))
#         image_array = np.repeat(image_array[..., np.newaxis], 3, axis=-1)
#         model.set_image(image_array)
#         model.calc_features()
#         return {"success": True}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference")
async def inference(sampled_points: List[Location] = [Location(x=0, y=0, z=0)]):
    global model, redis, ct_transform, spacing
    model, redis = check_model_and_redis(model, redis)
    try:
        sampled_points_np = np.array([[point.x, point.y, point.z] for point in sampled_points ])
        sampled_points_np = sampled_points_np @ ct_transform[:3, :3].T + ct_transform[:3, 3]
        sampled_points_np = sampled_points_np * spacing
        sampled_points_tensor = torch.tensor(sampled_points_np, dtype=torch.float32).to(device)
        sims = model.inference(sampled_points_tensor)
        pred_proj_point, max_vals = find_batch_peak_coordinates_maxval(sims)
        pred_proj_point = pred_proj_point.cpu().numpy()[0]
        # gt_proj_point = query_projection_matrix @ np.concatenate([sampled_points_np, np.ones((sampled_points_np.shape[0], 1))], axis=-1).T
        # gt_proj_point = gt_proj_point[:2, :] / gt_proj_point[2, :]
        # gt_proj_point = gt_proj_point.T[0] # (N, 2)
        # gt_proj_point = 224 - gt_proj_point
        heatmap = sims[0].cpu().numpy()
        return numpy_to_colormap_image_response(heatmap, pred_proj_point)
    except Exception as e:
        tb_info = traceback.format_exc()
        logger.error(tb_info)
        raise HTTPException(status_code=500, detail=str(e))

def check_model_and_redis(model, redis):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if redis is None:
        raise HTTPException(status_code=500, detail="Redis not loaded")
    return model, redis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)