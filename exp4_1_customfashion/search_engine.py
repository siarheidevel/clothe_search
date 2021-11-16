import torch
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor, NamedTupleCursor
import logging, time
from pathlib import Path
from collections import OrderedDict
import cv2
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms as T

CUDA_DEVICE_ID = 0
CUDA_DEVICE= 'cuda:'+str(CUDA_DEVICE_ID)
torch.cuda.set_device(CUDA_DEVICE)

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import model, dataset, utils


DSN = "dbname=postgres user=postgres password=111111 port=5433 host=127.0.0.1"

garment_model = None


def load_model():
    global garment_model

    garment_model = model.Sim_model(output_dim=64)
    garment_model.eval()

    emb_short, emb_full = garment_model(torch.randn(1,3,224,224))
    checkpoint  = torch.load('/home/deeplab/devel/Clothes_search/exp3_customfashion/checkpoints/dense121_category/epoch=19.ckpt',
        map_location=lambda storage, loc: storage)
    #load only model parameters, exclude 'model.' from naming
    model_dict = OrderedDict([(n[6:],d) for (n,d) in checkpoint['state_dict'].items() if n.startswith('model.')])
    garment_model.load_state_dict(model_dict)

    emb_short, emb_full = garment_model(torch.randn(1,3,224,224))
    return garment_model


def garment_vector(image_file, seg_file, seg_id: int, bg_color=(0,0,0), out_size=(224,224)):
    """
    return 3,224,224 tensor
    """
    img_rgb = cv2.imread(image_file)[:,:,[2,1,0]]#Image.fromarray(img_rgb.astype(np.uint8)).save('img_res.png')
    seg = np.array(Image.open(seg_file))
    mask = seg==seg_id
    bbox_img = utils.bounding_box(seg*mask)
    bg_color=(0,0,0)
    garment_img = np.ones_like(img_rgb) * bg_color
    garment_img[mask] = img_rgb[mask]
    garment_img = garment_img[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
    garment_img = utils.resize_to_box(garment_img,
        to_size=out_size, keep_aspect=True, bg_color=bg_color)
    tfms = T.Compose([T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    garment_tensor = tfms(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
    return garment_tensor #Image.fromarray(np.transpose((utils.denormalize_tensor(g_tensors[0])*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')


def fill_vector_data():
    # unfilled garments
    # fill garment vetcors from model
    global garment_model
    if garment_model is None:
        garment_model = load_model()
    # db_query('delete from garment_vector',update=True)
    from_garment_id=353800
    per_page = 500
    while True:
        start_time = time.time()
        g_rows = db_query(f'''
            select garment_id,seg_id,photo from garment where garment_id>{from_garment_id} 
            order by garment_id asc limit {per_page}
        ''')
        db_select_time = time.time()-start_time;start_time = time.time()
        # get tensors
        g_tensors = []
        for row in g_rows:
            g_tensors.append(garment_vector(row.photo,
                row.photo + '.seg_qanet.render.png', row.seg_id)[None,...])
        g_tensors = torch.cat(g_tensors) #Image.fromarray(np.transpose((utils.denormalize_tensor(g_tensors[0])*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
        preprocess_time = time.time()-start_time;start_time = time.time()
        # get embeddings
        with torch.no_grad():
            emb_short, emb_full = garment_model(g_tensors.to(CUDA_DEVICE))
        #save embeddings
        emb_short = emb_short.cpu().numpy().astype(np.float16)

        model_time = time.time()-start_time;start_time = time.time()



        with psycopg2.connect(DSN) as conn:
            with conn.cursor(cursor_factory=NamedTupleCursor) as cursor:
                args_str = ','.join([cursor.mogrify("(%s,%s)", (row.garment_id, emb_short[index].tolist())).decode('utf-8') for index,row in enumerate(g_rows)])
                cursor.execute("insert into garment_vector(garment_id, vecdata) VALUES " + args_str) 
        
        # for index,row in enumerate(g_rows):
        #     db_query(f'''
        #         insert into garment_vector(garment_id, vecdata) 
        #         values (%s,%s)
        #     ''', [row.garment_id, emb_short[index].tolist()], update=True)
        
        db_insert_time = time.time()-start_time;start_time = time.time()

        from_garment_id = row.garment_id
        print(f'{from_garment_id} select={db_select_time:.3f}s preprocess={preprocess_time:.3f}s model={model_time:.3f}s insert={db_insert_time:.3f}s')

        if len(g_rows) < per_page:
            break

SEG_ID_2_TYPE = {1:'hat', 4:'upper', 5:'bottom', 8:'shoes', 10:'gloves'}


def search_similar_to_id(garment_id:int):
    rows = db_query(f'''
    with req as(
        select g.garment_id,g.seg_id,g.photo,v.vecdata from garment g
        inner join garment_vector v on g.garment_id=v.garment_id
        where g.garment_id={garment_id} limit 1
    )
    select g.garment_id,g.seg_id,g.photo, (select req.vecdata from req) <-> v.vecdata as distance
    from garment_vector v
    inner join garment g on g.garment_id=v.garment_id
    where (select req.vecdata from req) <-> v.vecdata<0.3 
    order by v.vecdata <-> (select req.vecdata from req)  limit 10
    ''')

    images = []
    for r in rows:
        print(f'{r.distance:.4f} {r.seg_id} {r.garment_id} {r.photo}')
        img_rgb = cv2.imread(r.photo)[:,:,[2,1,0]] #from PIL import Image; Image.fromarray(img_rgb).save('img.jpg')
        # add text
        pil_im = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        draw.text((0, 0), f"{r.distance:.3f} {SEG_ID_2_TYPE.get(r.seg_id,'unknown')} {r.garment_id}", font=font, fill= (127,10,10))
        img_rgb = np.array(pil_im)


        img_rgb = utils.resize_to_box(img_rgb,
            to_size=(512,384), keep_aspect=True, bg_color=(255,255,255))
        images.append(img_rgb)
    return images


def search_similar_to_image(image_file, segment_file, seg_id:int, limit =10):
    start_time = time.time()
    g_tensor = garment_vector(image_file,
                segment_file, seg_id)[None,...]
    preprocess_time = time.time()-start_time;start_time = time.time()
    # get embeddings
    with torch.no_grad():
        emb_short, emb_full = garment_model(g_tensor)
    #save embeddings
    emb_short = emb_short.cpu().numpy().astype(np.float16)[0].tolist()
    model_time = time.time()-start_time;start_time = time.time()

    rows = db_query(f"""
    select g.garment_id,g.seg_id,g.photo, v.vecdata <-> '{emb_short}' as distance
    from garment_vector v
    inner join garment g on g.garment_id=v.garment_id
    where v.vecdata <-> '{emb_short}' < 0.3 
    order by v.vecdata <-> '{emb_short}'  limit {limit}
    """)

    images = []
    images.append(utils.resize_to_box(cv2.imread(image_file)[:,:,[2,1,0]],
            to_size=(512,384), keep_aspect=True, bg_color=(255,255,255)))
    for r in rows:
        print(f'{r.distance:.4f} {r.seg_id} {r.garment_id} {r.photo}')
        img_rgb = cv2.imread(r.photo)[:,:,[2,1,0]] #from PIL import Image; Image.fromarray(img_rgb).save('img.jpg')
        # add text
        pil_im = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        draw.text((0, 0), f"{r.distance:.3f} {SEG_ID_2_TYPE.get(r.seg_id,'unknown')} {r.garment_id}", font=font, fill= (127,10,10))
        img_rgb = np.array(pil_im)


        img_rgb = utils.resize_to_box(img_rgb,
            to_size=(512,384), keep_aspect=True, bg_color=(255,255,255))
        images.append(img_rgb)
    return images



def fill_init_data():
    """
     read csv and fill the db
    """
    data_file = '/home/deeplab/devel/Clothes_search/util/customfashion_index_qanet_processed2.csv'
    data = pd.read_csv(data_file, sep=';')
    data['garment_id'] = data.index

    # fill web page table
    page_df = data[['label_id','label','brand','orig_category','gender']].drop_duplicates()
    garment_df = data[['garment_id','label_id','seg_id','image_file']]

    # with psycopg2.connect(DSN) as conn:
    #     with conn.cursor(cursor_factory=NamedTupleCursor) as cursor:
    #         cursor.execute('delete from webpage')
    #         for row in page_df.itertuples():                
    #             cursor.execute(f'insert into webpage(page_id,name,url,category,gender) values(%s, %s,%s,%s,%s)',
    #                 [row.label_id,','.join([row.label,row.orig_category, row.brand]), row.label, row.orig_category, row.gender])

    # with psycopg2.connect(DSN) as conn:
    #     with conn.cursor(cursor_factory=NamedTupleCursor) as cursor:
    #         cursor.execute('delete from garment')
    #         for row in garment_df.itertuples():                
    #             cursor.execute(f'insert into garment(garment_id,page_id,seg_id,photo) values(%s,%s,%s,%s)',
    #                 [row.garment_id, row.label_id, row.seg_id, row.image_file])




    
    # cursor.execute('delete from garment')
    #             cursor.execute(f'insert into garment(garment_id,page_id,seg_id,photo', params)
    print(data.head(5))






def db_query(select_query: str, params=None, update: bool = False):
    logging.info(f'quering "{select_query}":{params}')
    with psycopg2.connect(DSN) as conn:
        with conn.cursor(cursor_factory=NamedTupleCursor) as cursor:
            cursor.execute(select_query, params)
            # print(cursor.description)
            if not update:
                return cursor.fetchall()

if __name__=='__main__':
    print(db_query('select 1+1'))
    print('')
    # fill_init_data()
    garment_model = load_model()
    garment_model.to(CUDA_DEVICE)
    
    print(garment_model)
    # fill_vector_data()


