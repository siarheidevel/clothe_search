from typing import Optional, List
from enum import Enum
import time
import uvicorn
from fastapi import FastAPI, Query, Path, Cookie, Header, File, UploadFile, Depends, Request, Response
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from PIL import Image

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

class Result(BaseModel):
    model: str
    description: Optional[str] = None
    x: List[int]
    y: List[int]
    height: int
    width: int

class ModelType(str, Enum):
    resnet = 'resnet'
    mobilenet = 'mobilenet'
    vgg = 'vgg19'


# http://127.0.0.1:8000/redoc
# http://127.0.0.1:8000/docs

app = FastAPI(debug=True)

@app.get("/models/{model_type}", response_model=Result, response_model_exclude={"height"})
async def get_model(model_type: ModelType,
    q: Optional[str] = Query(None, max_length=50, alias="item-query")):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    result = Result(model=model_type,x=[5,2,3], y=[4,5,6], height=50, width=100,
        description='Привет')
    # return result.dict()
    return result


@app.post("/models/update")
async def update_model(q:str = Query(..., max_length=50), 
    ads_id: str = Cookie(None, alias='ads cookeie')):
    return await get_model(ModelType.resnet, q)


class Item(BaseModel):
    name: str = Field(..., example="Foo")
    description: Optional[str] = Field(None, example="A very nice Item")
    price: float = Field(..., example=35.4)
    tax: Optional[float] = Field(None, example=3.2)


@app.get("/headers/")
async def read_items(user_agent: Optional[str] = Header(None)):
    return {"User-Agent": user_agent}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/items/", tags=['secure'])
async def read_items_secure(token: str = Depends(oauth2_scheme)):
    return {"token": token}


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results


@app.post("/files/", tags=["files"])
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/", tags=["files"])
async def create_upload_file(file: UploadFile = File(...)):
    # contents = await file.read()
    
    # import io
    # img = Image.open(io.BytesIO(contents))
    img = Image.open(file.file)
    return {"filename": file.filename,"size":img.size}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.post("/predict/segment")
def predict(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    image = Image.open(io.BytesIO(file_bytes))
    new_image = prepare_image(image)
    result = predict(image)
    bytes_image = io.BytesIO()
    new_image.save(bytes_image, format='PNG')
    return Response(content = bytes_image.getvalue(), headers = result, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run('web_api:app', host="0.0.0.0", port=8000, reload=True)

