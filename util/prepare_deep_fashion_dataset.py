import numpy as np
from PIL import Image
from numpy.lib.function_base import append
import pandas as pd
import glob, os, re, logging, sys
from pathlib import Path
from pandas.core.frame import DataFrame
logging.basicConfig(level = logging.INFO)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
#                'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8, 'skin':9, 'glove':10 }
seg_labels = {'hat':1, 'upper-clothes':4, 'pants': 5, 'shoes':8, 'glove':10 }

deepfashion_tops =set(['Tees_Tanks', 'Blouses_Shirts', 'Dresses', 'Sweaters',
        'Jackets_Coats', 'Sweatshirts_Hoodies', 'Rompers_Jumpsuits', 'Cardigans',
        'Graphic_Tees', 'Shirts_Polos', 'Jackets_Vests', 'Suiting'])

deepfashion_bottoms = set([ 'Shorts', 'Pants', 'Skirts',
        'Denim', 'Leggings', 'Suiting'])

category_labels ={**{k:seg_labels['upper-clothes'] for k in deepfashion_tops},
        **{k:seg_labels['pants'] for k in deepfashion_bottoms}}


def parse_dir(dir_path: str):
    seg_file_list = glob.glob(dir_path + '/**/*.seg_qanet.render.png', recursive=True)
    garments = []
    for index, seg_file in enumerate(seg_file_list):        
        img_file = seg_file[:-len('.seg_qanet.render.png')]
        if not Path(img_file).exists():
            continue
        if index % 1000 == 0:
            logging.warning(f"{index}/{len(seg_file_list)}  {img_file}")
        try:
            title_search = re.search(r'/(WOMEN|MEN)/(.+)/(id_\d{8})/(\d{2})', img_file, re.IGNORECASE)
            if title_search:
                gender_str = title_search.group(1)
                category = title_search.group(2)
                label = title_search.group(3)+'_'+title_search.group(4)

                maybe_seg_id = category_labels.get(category, 0)

                seg_threshold = 20*20
                seg = np.array(Image.open(seg_file))
                for seg_label, seg_id in seg_labels.items():
                    total_count =np.sum((seg==seg_id).astype(np.uint8))
                    maybe_category = seg_label
                    if seg_id == maybe_seg_id:
                        maybe_category = category + '_' + seg_label
                    if total_count > seg_threshold:                        
                        garments.append(
                            [img_file, label+'_'+str(seg_id), seg_id, gender_str, maybe_category]
                        )
                    else:
                        pass       
                
        except Exception as e:
            logging.warning("Exception",e)

    garments_pd = pd.DataFrame(data=garments,
        columns=['image_file', 'label', 'seg_id', 'gender', 'maybe_category'])
    
    logging.info(f'total {len(garments_pd)} from {len(seg_file_list)} files')
    logging.info(f'uniq {garments_pd.label.value_counts().count()}, mean per label {garments_pd.label.value_counts().mean()}')
    logging.info(garments_pd[['gender','maybe_category']].value_counts())
    garments_pd.to_csv('deepfashion_index_qanet.csv', index=False, sep=';')
    return garments_pd


if __name__=="__main__":
    parse_dir(dir_path='/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres/')
    logging.info('finished')
