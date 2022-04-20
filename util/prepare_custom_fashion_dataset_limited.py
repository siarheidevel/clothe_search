import numpy as np
from PIL import Image
from numpy.lib.function_base import append
import pandas as pd
import glob, os, re, logging, sys, json
from pathlib import Path
from pandas.core.frame import DataFrame
logging.basicConfig(level = logging.INFO)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
#                'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8, 'skin':9, 'glove':10 }
seg_labels = {'hat':1, 'upper-clothes':4, 'pants': 5, 'shoes':8, 'glove':10 }
segs_labels = {'h':1, 't':4, 'b': 5, 's':8, 'g':10 }

deepfashion_tops =set(['Tees_Tanks', 'Blouses_Shirts', 'Dresses', 'Sweaters',
        'Jackets_Coats', 'Sweatshirts_Hoodies', 'Rompers_Jumpsuits', 'Cardigans',
        'Graphic_Tees', 'Shirts_Polos', 'Jackets_Vests', 'Suiting'])

deepfashion_bottoms = set([ 'Shorts', 'Pants', 'Skirts',
        'Denim', 'Leggings', 'Suiting'])

category_labels ={**{k:seg_labels['upper-clothes'] for k in deepfashion_tops},
        **{k:seg_labels['pants'] for k in deepfashion_bottoms}}

'''
{
  "brand": "Erolanta",
  "commentsCount": 15,
  "gender": "woman",
  "id": 9844672,
  "img_src": "https://images.wbstatic.net/large/new/9840000/9844672-1.jpg",
  "mark": 4,
  "name": "Комбинация",
  "picsCnt": 6
}

{
  "brand": "adidas Combat",
  "category": "Шорты",
  "color": "синий",
  "gallery_list": [
    "//a.lmcdn.ru/img236x341/A/D/AD015EMBOIC9_6718027_1_v1.jpg",
    "//a.lmcdn.ru/img236x341/A/D/AD015EMBOIC9_6718028_2_v1.jpg",
    "//a.lmcdn.ru/img236x341/A/D/AD015EMBOIC9_6718029_3_v1.jpg"
  ],
  "gender": "unisex",
  "id": "AD015EMBOIC9",
  "name": "Шорты спортивные"
}
'''
def parse_dir(dir_path: str):
    garments=[]
    json_file_list = glob.glob(dir_path + '/**/meta.json', recursive=True)
    for index, meta_file in enumerate(json_file_list):
        if index % 1000 == 0:
            logging.warning(f"{index}/{len(json_file_list)}  {meta_file}")
        try:
            with open(meta_file,'r') as mf:
                meta = json.load(mf)
            id = meta['id']
            gender = meta.get('gender', None)
            category = meta['name']
            brand = meta.get('brand', None)

            # list  files
            for seg_file in glob.glob(os.path.dirname(meta_file) + '/*.seg_qanet.render.png', recursive=False):
                img_file = seg_file[:-len('.seg_qanet.render.png')]
                seg_threshold = 40*40
                seg = np.array(Image.open(seg_file))

                for seg_label, seg_id in seg_labels.items():
                    total_count =np.sum((seg==seg_id).astype(np.uint8))
                    if total_count > seg_threshold:                        
                        garments.append(
                            [img_file, id, category, seg_id, gender, brand, seg_label]
                        )
                    else:
                        pass          

        except Exception as e:
            logging.warning("Exception",e)

    garments_pd = pd.DataFrame(data=garments,
        columns=['image_file', 'label', 'category','seg_id', 'gender', 'brand', 'maybe_category'])
    
    logging.info(f'total {len(garments_pd)} from {len(json_file_list)} files')
    logging.info(f'uniq {garments_pd.label.value_counts().count()}, mean per label {garments_pd.label.value_counts().mean()}')
    logging.info(garments_pd[['category','gender','maybe_category']].value_counts())
    garments_pd.to_csv('customfashion_index_qanet.csv', index=False, sep=';')
    return garments_pd

def preprocess_data(csv_file = 'customfashion_index_qanet.csv'):
    data = pd.read_csv(csv_file, sep=';')
    # data = data.sample(frac=1, random_state=42)
    # TODO split data
    data=data.fillna('unknown')
    # data['orig_category']=data.category
    data['gender'] = data.apply(lambda  x: {'women':'woman','men':'man',
        'girls':'children','girl':'children','boys':'children','unisex':'children',
        }.get(x.gender,x.gender), axis=1)
    # remove gender and category dublicates
    data['category_short'] = data.apply(lambda x:re.split(r'\W+',x.category.lower())[0][:], axis=1 )
    # categories_df.apply(lambda x:x.category.lower().split()[0][:], axis=1 ).unique()


    # # data['category'].value_counts().to_csv('customfashion_categories_to_seg.csv', encoding='utf-8', index=True,sep='\t')
    # categories_df = data['category'].value_counts().reset_index().rename(columns={'index':'category','category':'counts'})
    # categories_df['category_short']=categories_df.apply(lambda x:x.category.lower().split()[0][:], axis=1)
    # # sorted(categories_df.apply(lambda x:re.split(r'\W+',x.category.lower())[0][:], axis=1 ).unique().tolist())
    # categories_df['category_short']=categories_df.apply(lambda x:re.split(r'\W+',x.category.lower())[0][:], axis=1 )
    # categories_df.to_csv('customfashion_categories_to_seg.csv', encoding='utf-8', index=False,sep='\t')

    # categorries 2 segs
    categories_df = pd.read_csv('customfashion_categories_to_seg.csv', sep='\t', names=['category','counts','category_short','segs'])
    categories_df.groupby('segs').sum().sort_values('counts', ascending=False)[:20]
    categories_df.groupby(['category_short','segs']).sum().sort_values('counts', ascending=False)[:20]

    fitted_categories_df = categories_df.groupby(['category_short','segs']).sum().reset_index()
    fitted_categories_df['segs'] = fitted_categories_df['segs'].str.split(',')
    fitted_categories_df=fitted_categories_df.explode('segs')
    fitted_categories_df['seg_id']=fitted_categories_df.segs.apply(lambda x:segs_labels.get(x,-1))

    data = pd.merge(data,fitted_categories_df,how='inner', on=['category_short','seg_id'])
    



    top_labels = pd.DataFrame(data[['label','seg_id','category_short']].value_counts().reset_index().to_numpy(),
        columns=['label', 'seg_id','category_short', 'cnt'])
    # remove rows where label count<3 
    # https://stackoverflow.com/questions/44706485/how-to-remove-rows-in-a-pandas-dataframe-if-the-same-row-exists-in-another-dataf
    data = pd.merge(data, top_labels[top_labels.cnt<3], indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    data.drop(columns=['cnt'],inplace=True)
    label_stat = pd.DataFrame(data[['label', 'category_short', 'seg_id']].value_counts().reset_index().to_numpy(),
        columns=['label', 'category_short', 'seg_id', 'label_count'])
    label_stat['label_weight']=label_stat.label_count.sum()/label_stat.label_count
    label_stat['label_id'] = np.arange(label_stat.shape[0])
    data = data.merge(label_stat,how='inner', on=['label', 'category_short', 'seg_id'])

    gender_stat = pd.DataFrame(data.gender.value_counts().reset_index().to_numpy(),
        columns=['gender', 'gender_count'])
    gender_stat['gender_weight']=gender_stat.gender_count.sum()/gender_stat.gender_count
    gender_stat['gender_id'] = np.arange(gender_stat.shape[0])
    data = data.merge(gender_stat,how='inner', on='gender')

    # category_stat = pd.DataFrame(data.category.value_counts().reset_index().to_numpy(),
    #     columns=['category', 'category_count'])
    # category_stat['category_weight']=np.log((category_stat.category_count.sum()/category_stat.category_count).to_numpy().astype(np.float))
    # category_stat['category_id'] = np.arange(category_stat.shape[0])
    # data = data.merge(category_stat,how='inner', on='category')

    # seg_stat = pd.DataFrame(data.seg_id.value_counts().reset_index().to_numpy(),
    #     columns=['seg_id', 'seg_count'])
    # seg_stat['seg_weight']=np.log((seg_stat.seg_count.sum()/seg_stat.seg_count).to_numpy().astype(np.float))   
    # data = data.merge(seg_stat,how='inner', on='seg_id')

    category_seg_stat = pd.DataFrame(data[['category_short', 'seg_id']].value_counts().reset_index().to_numpy(),
        columns=['category_short','seg_id', 'category_count'])
    category_seg_stat['category_weight']=np.log((category_seg_stat.category_count.sum()/category_seg_stat.category_count).to_numpy().astype(np.float))
    category_seg_stat['category_id'] = np.arange(category_seg_stat.shape[0])
    data = data.merge(category_seg_stat,how='inner', on=['category_short','seg_id'])

    # normaise by max
    data.label_weight=data.label_weight/data.label_weight.max()
    data.category_weight=data.category_weight/data.category_weight.max()
    data.gender_weight=data.gender_weight/data.gender_weight.max()
    # split train and test data
    data.to_csv('customfashion_index_qanet_processed_limited_one_seg.csv', index=False, sep=';')

    category_count = len(data.category_id.value_counts())
    instance_count = len(data.label_id.value_counts())


def preprocess_data_one_category(csv_file = 'customfashion_index_qanet.csv'):
    data = pd.read_csv(csv_file, sep=';')
    # data = data.sample(frac=1, random_state=42)
    # TODO split data
    data=data.fillna('unknown')
    # data['orig_category']=data.category
    data['gender'] = data.apply(lambda  x: {'women':'woman','men':'man',
        'girls':'children','girl':'children','boys':'children','unisex':'children',
        }.get(x.gender,x.gender), axis=1)
    # remove gender and category dublicates
    data['category_short'] = data.apply(lambda x:re.split(r'\W+',x.category.lower())[0][:], axis=1 )

    categories_df = pd.read_csv('category2seg.csv', sep='\t', names=['category_short','segs'])
    categories_df['seg_id'] = categories_df.segs.apply(lambda x:[segs_labels.get(t,-1) for t in x.split(',')])
    categories_df=categories_df.explode('seg_id')

    # merge data by category  and seg_id
    data_limited = data.merge(categories_df, how='inner', on=['category_short','seg_id'])

    # group by photo
    data_limited = data_limited.groupby(['image_file','label','gender','category','category_short','segs'],as_index = False).agg({'seg_id': list})

    # remove where label instances less than 3
    top_labels = pd.DataFrame(data_limited[['label']].value_counts().reset_index().to_numpy(),
        columns=['label', 'cnt'])
    data_limited = pd.merge(data_limited, top_labels[top_labels.cnt<3], indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    data_limited.drop(columns=['cnt'],inplace=True)

    # build label_id
    label_stat = pd.DataFrame(data_limited[['label']].value_counts().reset_index().to_numpy(),
        columns=['label','label_count'])
    label_stat['label_weight']=label_stat.label_count.sum()/label_stat.label_count
    label_stat['label_id'] = np.arange(label_stat.shape[0])
    data_limited = data_limited.merge(label_stat,how='inner', on=['label'])

    # build category_id
    category_seg_stat = pd.DataFrame(data_limited[['category_short']].value_counts().reset_index().to_numpy(),
        columns=['category_short','category_count'])
    category_seg_stat['category_weight']=np.log((category_seg_stat.category_count.sum()/category_seg_stat.category_count).to_numpy().astype(np.float))
    category_seg_stat['category_id'] = np.arange(category_seg_stat.shape[0])
    data_limited = data_limited.merge(category_seg_stat,how='inner', on=['category_short'])

    # build gender_id
    gender_stat = pd.DataFrame(data_limited.gender.value_counts().reset_index().to_numpy(),
        columns=['gender', 'gender_count'])
    gender_stat['gender_weight']=gender_stat.gender_count.sum()/gender_stat.gender_count
    gender_stat['gender_id'] = np.arange(gender_stat.shape[0])
    data_limited = data_limited.merge(gender_stat,how='inner', on='gender')

    # build category_gender_id
    category_gender_stat= pd.DataFrame(data_limited[['category_short','gender']].value_counts().reset_index().to_numpy(),
        columns=['category_short','gender','category_gender_count'])
    category_gender_stat['category_gender_weight']=category_gender_stat.category_gender_count.sum()/category_gender_stat.category_gender_count    
    category_gender_stat['category_gender_id'] = np.arange(category_gender_stat.shape[0])
    data_limited = data_limited.merge(category_gender_stat,how='inner', on=['category_short','gender'])
    
    # save data
    data_limited.to_csv('customfashion_index_qanet_processed_limited_one_category_milti_seg.csv', index=False, sep=';')


    



    print(categories_df)
    



if __name__=="__main__":
    # parse_dir(dir_path='/home/deeplab/datasets/custom_fashion/data/')
    preprocess_data_one_category('customfashion_index_qanet.csv')
    logging.info('finished')
  