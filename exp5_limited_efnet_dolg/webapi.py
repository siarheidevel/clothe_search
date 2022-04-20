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


@app.get("/search_id/")
def segment(response: Response, garment_id: int):
    """
    search similar garments by garment_id
    """
    images = se.search_similar_to_id(garment_id=garment_id)
    images = np.concatenate(images,axis=1)
    
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    new_image_file = os.path.join(TEMP_DIR, 
        ''.join(random.choice('abcdefgh') for i in range(4))+'_.jpg')
    Image.fromarray(images).save(new_image_file)
    return FileResponse(new_image_file, headers=response.headers, media_type='image/jpeg')

class GarmentType(IntEnum , Enum):
    '''
    [SEG.ID['hair'], SEG.ID['shoes'], SEG.ID['pants'], SEG.ID['upper-clothes'], SEG.ID['hat']]
    '''
    top = 5
    bottom = 4
    hat =1
    shoes = 8

@app.post("/search/")
def search(response: Response, seg_id:GarmentType = GarmentType.top,
    image_file: UploadFile = File(...)):
    """
    Search simiar garment
    """
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    new_image_file = os.path.join(TEMP_DIR, 
        ''.join(random.choice('wyxzrst') for i in range(4))+'_'+ image_file.filename)
    with open(new_image_file, "wb+") as file_object:
        shutil.copyfileobj(image_file.file, file_object)
    
    # get parsing result
    #     curl -X 'POST' \
    #   'http://localhost:8010/segment/?render=true' \
    #   -H 'accept: application/json' \
    #   -H 'Content-Type: multipart/form-data' \
    #   -F 'upload_file=@02_1_front.jpg;type=image/jpeg'
    with open(new_image_file, 'rb') as f:
        files = {'upload_file': f}
        response = requests.post('http://localhost:8010/segment/?render=true', files=files)
        if response.status_code == 200:
            with open(new_image_file+'.seg_qanet.render.png', "wb") as file:
                file.write(response.content)
    images = se.search_similar_to_image(new_image_file,
        new_image_file+'.seg_qanet.render.png', seg_id, limit=8)
    images = np.concatenate(images,axis=1)
    Image.fromarray(images).save(new_image_file+'.result.jpg')
    return FileResponse(new_image_file+'.result.jpg',  media_type='image/jpeg')
    




if __name__ == "__main__":
    uvicorn.run('webapi:app', host="0.0.0.0", port=8040, reload=True, workers =1)