import uvicorn
from fastapi import FastAPI, Query, Path, Cookie, Header, File, UploadFile, Depends, Request, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
import logging, time, random, shutil, requests
from pathlib import Path
from PIL import Image
from io import StringIO, BytesIO
import numpy as np
from enum import Enum, IntEnum

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import search_engine as se

TEMP_DIR = '/tmp/model/garment_search'

app = FastAPI(debug=False)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.on_event("startup")
async def startup_event():
    logging.info('Loading model')
    se.load_model()
    # segment_processor.parse_image(model,'/home/deeplab/datasets/custom_fashion/demo_/1544/15445714/15445714-1.jpg')
    logging.info('Loaded model')


@app.post("/vector/")
def search(image_file: UploadFile = File(...)):
    """
    Search simiar garment
    """    
    embedding = se.garment_vector(image_file.file)
    
    return {'vector':embedding.tolist()}
    

if __name__ == "__main__":
    uvicorn.run('webapi:app', host="0.0.0.0", port=8025, reload=True, workers =1)