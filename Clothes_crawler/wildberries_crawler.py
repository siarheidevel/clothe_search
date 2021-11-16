# ["//a.lmcdn.ru/img236x341/L/E/LE306EWFOWD5_8853255_1_v2_2x.jpg" , 
# "//a.lmcdn.ru/img236x341/L/E/LE306EWFOWD5_8853256_2_v2_2x.jpg" , 
# "//a.lmcdn.ru/img236x341/L/E/LE306EWFOWD5_8853257_3_v2_2x.jpg"]

# https://a.lmcdn.ru/img389x562/R/T/RTLAAK930201_14522509_1_v1_2x.jpg
# https://a.lmcdn.ru/img236x341/L/E/LE306EWFOWD5_8853255_1_v2_2x.jpg
# https://a.lmcdn.ru/img389x562/L/E/LE306EWFOWD5_8853255_1_v2_2x.jpg
# https://a.lmcdn.ru/img600x866/V/I/VI004EWLVIA8_13046310_3_v1.jpg
# https://a.lmcdn.ru/product/V/I/VI004EWLVIA8_13046309_2_v1.jpg

import requests
from bs4 import BeautifulSoup
import json
import re
import shutil
from pathlib import Path
import pandas as pd
import logging
import datetime;
import time
from itertools import cycle
from requests.models import parse_url

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# proxies
# https://www.scrapehero.com/how-to-rotate-proxies-and-ip-addresses-using-python-3/

headers = {
    'authority': 'www.wildberries.ru',
    'cache-control': 'max-age=0',
    'upgrade-insecure-requests': '1',
    # 'user-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36'
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36',
    'sec-fetch-dest': 'document',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'accept-encoding': 'gzip, deflate, br',
    'cookie': 'route=d9844383890dded2001a7340cfb01b55a300c8c2; BasketUID=b2522f27-82ef-4337-b4ab-cdf5f79be4e1; ___wbu=6ec38c6f-7617-493a-84de-529ef8e8fba3.1626436242; _wbauid=2942058101626436243; _gcl_au=1.1.1522023271.1626436244; _ga=GA1.2.2000920736.1626436244; _gid=GA1.2.1015732027.1626436244; __catalogOptions=Sort%3APopular%26CardSize%3Ac246x328; __wbl=cityId%3D77%26regionId%3D0%26city%3D%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B0%26phone%3D88001007505%26latitude%3D55%2C755798%26longitude%3D37%2C617599%26src%3D1; __store=119261_122252_122256_117673_122258_122259_121631_122466_122467_122495_122496_122498_122590_122591_122592_123816_123817_123818_123820_123821_123822_124093_124094_124095_124096_124097_124098_124099_124100_124101_124583_124584_127466_126679_126680_127014_126675_126670_126667_125186_125611_116433_6159_507_3158_117501_120762_119400_120602_6158_121709_1699_2737_117986_1733_686_117413_119070_118106_119781; __region=64_75_4_38_30_33_70_1_71_22_31_66_80_69_48_40_68; __pricemargin=1.0--; __cpns=12_3_18_15_21; __sppfix=; __spp=0; ___wbs=4fb823b7-0e41-41b0-b323-da38ccb4fbb8.1626521845; ncache=119261_122252_122256_117673_122258_122259_121631_122466_122467_122495_122496_122498_122590_122591_122592_123816_123817_123818_123820_123821_123822_124093_124094_124095_124096_124097_124098_124099_124100_124101_124583_124584_127466_126679_126680_127014_126675_126670_126667_125186_125611_116433_6159_507_3158_117501_120762_119400_120602_6158_121709_1699_2737_117986_1733_686_117413_119070_118106_119781%3B64_75_4_38_30_33_70_1_71_22_31_66_80_69_48_40_68%3B1.0--%3B12_3_18_15_21%3B%3B0%3BSort%3APopular%26CardSize%3Ac246x328; criteo_uid=5_4YbdZcLysDAduTqVI3ZpW9FF9UX62N; _dc_gtm_UA-2093267-1=1; _pk_ref.1.034e=%5B%22%22%2C%22%22%2C1626521847%2C%22https%3A%2F%2Fwww.google.com%2F%22%5D; _pk_ses.1.034e=*; _pk_id.1.034e=73f8b19d8a5bb412.1626436245.5.1626521891.1626521847.',
    'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"'
    

}

# session = requests.session()

# url = 'https://www.wildberries.ru/catalog/zhenshchinam/odezhda/verhnyaya-odezhda?sort=popular&page=6'
# response = session.get(url, headers=headers)
# if response.status_code !=200:
#     logging.warning(f'Bad code={response.status_code} for {url}')
# soup = BeautifulSoup(response.text, 'html.parser')

# # loading json data
# json_products = None
# for index,script in enumerate(soup.find_all("script")):
#     script_text=script.string
#     if script_text is None or len(script_text)< 100: continue
#     if len(script_text)>10000: print(index)
#     json_module =  re.search(r"ssrModel:(.+),\n",script_text) 
#     if json_module is not None:
#         json_module=json.loads(json_module.group(1))
#         json_products = {p['cod1S']:p for p in json_module['model']['products']}
#         break

# garment_list = soup.find_all("div",{"class":"j-card-item"})
# for index, item in enumerate(garment_list):
#     id = int(item["data-catalogercod1s"])
#     product = json_products[id]
#     # name=item.select_one('.goods-name').text
#     # brand=item.select_one('.brand-name').text
#     name = product['name']
#     color = product['color']
#     brand = product['brand']
#     adult = product['adult']
#     mark =product['mark']
#     picsCnt = product['picsCnt']
#     img_src=item.select_one('img[src]').get('src')

#     print('dd')

#     print(script.text[:1000])

proxy_data ='''
	
169.57.1.85	80
158.69.25.178	32769
52.143.130.19	3128
169.57.1.84	80
159.8.114.34	8123
159.8.114.34	25
159.8.114.37	80
159.8.114.37	8123
159.8.114.37	25
51.222.21.93	32768
169.57.1.84	8123
137.74.245.212	43567
169.57.1.85	8123
95.216.194.46	1080
79.175.176.251	808
188.120.250.142	8080
51.68.207.81	80



'''

proxies = [l.split("\t")[0]+':'+l.split("\t")[1] for l in proxy_data.splitlines() if len(l)>10]
proxy_pool = cycle(proxies)

class Crawler:
    DATA_DIR = Path('/home/deeplab/datasets/custom_fashion/data/wildberries_ru')

    def __init__(self) -> None:
        self.to_parse_list = set()
        self.parsed_list = {}

    def _set_parsed_page(self, url):
        self.parsed_list[url] = datetime.datetime.now
        if url in self.to_parse_list:
            self.to_parse_list.remove(url)

    def _add_page_for_parsing(self, url):
        if url not in self.parsed_list:
            self.to_parse_list.add(url)

    def _next_page_to_parse(self) -> str:
        try:
            return self.to_parse_list.pop()
        except KeyError:
            return None


    @classmethod
    def start_crawling(cls, init_urls: list):
        crawler = cls()
        crawler.to_parse_list.update(init_urls)
        parse_url =  crawler._next_page_to_parse()
        while parse_url is not None:
            try:
                crawler.parse_page(parse_url)
            except Exception as e:
                logging.exception(f'Error parsing {parse_url}: {e}')        
            parse_url =  crawler._next_page_to_parse()
    def _get_session(self):
        # if hasattr(self, 'session') and self.session is not None:
        #     return self.session
        self.session =  requests.session()
        return self.session

 
    def parse_page(self, url):
        # RESOLUTION_PATH = {'big', 'large'}
        # img_height_option = RESOLUTION_PATH['800']

        session =self._get_session()
        time.sleep(5)
        proxy = next(proxy_pool)
        # response = session.get(url, headers=headers,proxies={"http": proxy, "https": proxy})
        response = session.get(url, headers=headers)
        if response.status_code !=200:
            logging.warning(f'Bad code={response.status_code} for {url}')
        soup = BeautifulSoup(response.text, 'html.parser')

        # loading json data
        json_products = None
        for index,script in enumerate(soup.find_all("script")):
            script_text=script.string
            if script_text is None or len(script_text)< 100: continue
            # if len(script_text)>10000: print(index)
            json_module =  re.search(r"ssrModel:(.+),\n",script_text) 
            if json_module is not None:
                json_module=json.loads(json_module.group(1))
                json_products = {p['nmId']:p for p in json_module['model']['products']}
                break
        if json_products is None:
            logging.warning(f'No json data for {url}')
            return

        for c in response.cookies:
            if c.name == '__region':
                regions = c.value.replace('_',',')
            elif c.name == '__store':
                stores= c.value.replace('_',',')
        
        xQuery = json_module['model']['xData']['xcatalogQuery']
        xCatalog = json_module['model']['xData']['xcatalogShard']
        page=re.search(r'page=(\d+)',url).group(1)
        ajax_url = f'https://wbxcatalog-ru.wildberries.ru/{xCatalog}/catalog?spp=0&regions={regions}&stores={stores}&pricemarginCoeff=1.0&reg=0&appType=1&offlineBonus=0&onlineBonus=0&emp=0&locale=ru&lang=ru&curr=rub&couponsGeo=12,3,18,15,21&{xQuery}&sort=newly&page={page}'
        ajax_resp = session.get(ajax_url, headers=headers,cookies=response.cookies)
        if ajax_resp.status_code !=200:
            logging.warning(f'Bad code={ajax_resp.status_code} for {ajax_url}')

        ajax_data = json.loads(ajax_resp.text)

        for index,product in enumerate(ajax_data['data']['products']):
            try:
                meta = {}
                id =int(product['id']) 
                src = f'https://images.wbstatic.net/large/new/{id//10000 *10000}/{id}-1.jpg'
                picsCnt = product['pics']
                if 'zhenshchinam' in url:
                    meta['gender']='woman'
                elif 'muzhchinam' in url:
                    meta['gender']='man'
                elif 'dlya-devochek' in url:
                    meta['gender']='girl'
                elif 'muzhchinam' in url:
                    meta['dlya-malchikov']='boy'    
                try:
                    meta['name'] = product['name']
                except Exception: logging.warning(f' no name: id={id} , url={url} , index={index}')
                try:
                    meta['brand'] = product['brand']
                except Exception: logging.warning(f' no brand: id={id} , url={url} , index={index}')
                try:
                    meta['mark'] = product['rating']
                except Exception: logging.warning(f' no mark: id={id} , url={url} , index={index}')
                try:
                    meta['commentsCount'] = product['feedbacks']
                except Exception: logging.warning(f' no commentsCount: id={id} , url={url} , index={index}')
                
                
                meta['id']=id
                meta['img_src'] = src
                meta['picsCnt'] = picsCnt
                # '//images.wbstatic.net/c246x328/new/13390000/13394475-1.jpg'

                headers['referer'] = url
                garment_dir = Path(self.DATA_DIR, str(id))
                garment_dir.mkdir(parents=True,exist_ok=True)

                for img_index in range(1,picsCnt+1):
                    try:
                        link_800 = src = f'https://images.wbstatic.net/large/new/{id//10000 *10000}/{id}-'+str(img_index)+'.jpg' 
                        image_filename = Path(garment_dir , link_800[link_800.rindex('/')+1:])                
                        if not image_filename.exists() or image_filename.stat().st_size==0:
                            time.sleep(0.2)
                            img_resp = session.get(link_800, headers=headers, stream=True, cookies=response.cookies)                
                            if img_resp.status_code == 200:
                                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                                img_resp.raw.decode_content = True
                                # Open a local file with wb ( write binary ) permission.
                                with open(image_filename,'wb') as f:
                                    shutil.copyfileobj(img_resp.raw, f)
                                if image_filename.stat().st_size==0:
                                    logging.warning(f'Image dowload size is zero for {link_800}')
                                    return
                            else:
                                logging.warning(f'Image dowload bad code={img_resp.status_code} for {link_800}')
                        # else:
                        #     logging.warning(f'File exitst size={image_filename.stat().st_size} file={image_filename} for {link_800}')
                    except Exception as err:
                        logging.warning(f'Image dowload error for {link_800} ,{img_src},{img_index},{err} ')
                        
                json_file = Path(garment_dir , "meta.json")
                with open(json_file, 'w') as fp:
                    json_string = json.dumps(meta, default=lambda o: o.__dict__, sort_keys=True, indent=2, ensure_ascii=False)
                    fp.write(json_string)
            except:
                logging.warning(f' error parsing block: url={url} , index={index}')               
        
        self._set_parsed_page(url)





urls_config = '''
# man
https://www.wildberries.ru/catalog/muzhchinam/odezhda 500
https://www.wildberries.ru/catalog/muzhchinam/odezhda/bryuki-i-shorty 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/verhnyaya-odezhda 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/vodolazki 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/dzhempery-i-kardigany 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/dzhinsy 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/zhilety 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/karnavalnye-kostyumy 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/kigurumi 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/kombinezony 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/kostyumy 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/longslivy 10
https://www.wildberries.ru/catalog/muzhchinam/mantii 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/pidzhaki-i-zhakety 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/rubashki 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/svitshoty 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/tolstovki 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/futbolki-i-mayki 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/hudi 10
https://www.wildberries.ru/catalog/muzhchinam/odezhda/bryuki-i-shorty/shorty 10

https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery 150
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/bele-i-plavki 10
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/bryuki-i-dzhinsy 10
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/verhnyaya-odezhda 10
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/dzhempery-i-tolstovki 10
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/kostyumy-i-pidzhaki 10
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/rubashki 10
https://www.wildberries.ru/catalog/muzhchinam/bolshie-razmery/futbolki 10

https://www.wildberries.ru/catalog/muzhchinam/dlya-vysokih 10
https://www.wildberries.ru/catalog/muzhchinam/dlya-nevysokih 10


https://www.wildberries.ru/catalog/muzhchinam/bele 500


https://www.wildberries.ru/catalog/muzhchinam/odezhda/odezhda-dlya-doma 50

https://www.wildberries.ru/catalog/muzhchinam/ofis 100

https://www.wildberries.ru/catalog/muzhchinam/plyazhnay-odezhda 50

https://www.wildberries.ru/catalog/muzhchinam/svadba 40

https://www.wildberries.ru/catalog/muzhchinam/spetsodezhda 40
https://www.wildberries.ru/catalog/muzhchinam/spetsodezhda/meditsinskaya-odezhda 10
https://www.wildberries.ru/catalog/muzhchinam/spetsodezhda/rabochaya-odezhda 10
https://www.wildberries.ru/catalog/muzhchinam/spetsodezhda/golovnye-ubory 10



# womens
https://www.wildberries.ru/catalog/zhenshchinam/odezhda 500
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/bluzki-i-rubashki 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/bryuki-i-shorty 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/verhnyaya-odezhda 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/vodolazki 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/dzhempery-i-kardigany 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/dzhinsy-dzhegginsy 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/zhilety 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/karnavalnye-kostyumy 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/kombinezony-polukombinezony 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/kostyumy 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/longslivy 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/mantii 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/pidzhaki-i-zhakety 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/platya 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/svitshoty 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/tolstovki 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/tuniki 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/futbolki-i-topy 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/hudi 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/yubki 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/bryuki-i-shorty/shorty 10

https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery 500
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/bele 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/bluzki-rubashki-tuniki 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/verhnyaya-odezhda 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/dzhinsy-bryuki 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/zhilety 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/kostyumy 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/kupalniki 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/odezhda-dlya-doma 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/pidzhaki-zhakety 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/platya 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/pulovery-kofty-svitery 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/topy-futbolki 10
https://www.wildberries.ru/catalog/zhenshchinam/bolshie-razmery/yubki 10

https://www.wildberries.ru/catalog/zhenshchinam/dlya-vysokih 10
https://www.wildberries.ru/catalog/zhenshchinam/dlya-nevysokih 10


https://www.wildberries.ru/catalog/budushchie-mamy 300
https://www.wildberries.ru/catalog/budushchie-mamy/bluzki-koftochki 10
https://www.wildberries.ru/catalog/budushchie-mamy/bryuki-dzhinsy 10
https://www.wildberries.ru/catalog/budushchie-mamy/verhnyaya-odezhda 10
https://www.wildberries.ru/catalog/budushchie-mamy/kostyumy-kombinezony 10
https://www.wildberries.ru/catalog/budushchie-mamy/odezhda-dlya-doma 10
https://www.wildberries.ru/catalog/budushchie-mamy/pidzhaki-zhilety 10
https://www.wildberries.ru/catalog/budushchie-mamy/platya-sarafany 10
https://www.wildberries.ru/catalog/budushchie-mamy/pulovery-dzhempery 10
https://www.wildberries.ru/catalog/budushchie-mamy/topy-futbolki 10
https://www.wildberries.ru/catalog/budushchie-mamy/tuniki 10
https://www.wildberries.ru/catalog/budushchie-mamy/shorty-yubki 10


https://www.wildberries.ru/catalog/zhenshchinam/spetsodezhda 100
https://www.wildberries.ru/catalog/zhenshchinam/spetsodezhda/meditsinskaya-odezhda 10
https://www.wildberries.ru/catalog/zhenshchinam/spetsodezhda/rabochaya-odezhda 10


https://www.wildberries.ru/catalog/zhenshchinam/plyazhnaya-moda 200
https://www.wildberries.ru/catalog/zhenshchinam/kupalniki 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda-dlya-plyazha 10

https://www.wildberries.ru/catalog/svadba 150
https://www.wildberries.ru/catalog/zhenshchinam/svadba/platya-podruzhek-nevesty 10
https://www.wildberries.ru/catalog/zhenshchinam/svadba/svadebnye-platya 10


https://www.wildberries.ru/catalog/zhenshchinam/ofis 500
https://www.wildberries.ru/catalog/zhenshchinam/ofis/bluzki-i-rubashki 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/bryuki 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/dzhempery 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/kardigany 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/kostyumy 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/pidzhaki-i-zhilety 10
https://www.wildberries.ru/catalog/zhenshchinam/odezhda/ofisnye-platya 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/sarafany 10
https://www.wildberries.ru/catalog/zhenshchinam/ofis/yubki 10

https://www.wildberries.ru/catalog/zhenshchinam/odezhda/odezhda-dlya-doma 500
https://www.wildberries.ru/catalog/zhenshchinam/bele 500
https://www.wildberries.ru/catalog/zhenshchinam/bele/besshovnoe 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/bodi-i-korsety 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/byustgaltery 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/kolgotki-i-chulki 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/kombinatsii-i-neglizhe 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/korrektiruyushchee-bele 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/mayki 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/termobele 10
https://www.wildberries.ru/catalog/zhenshchinam/bele-i-kupalniki/trusy 10
# child
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/bryuki-i-shorty 100
https://www.wildberries.ru/catalog/detyam/dlya-malchikov/odezhda/verhnyaya-odezhda 100
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/vodolazki 20
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/dzhempery-i-kardigany 50
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/dzhinsy 20
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/zhilety 20
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/kostyumy 100
# https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/karnavalnye-kostyumy 20
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/odezhda-dlya-doma 50
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/plavki-i-bordshorty 20
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/rubashki 60
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-malchikov/futbolki-i-mayki 300


https://www.wildberries.ru/catalog/detyam/odezhda/dlya-devochek/bluzki-i-rubashki 50
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-devochek/bryuki-i-shorty 300
https://www.wildberries.ru/catalog/detyam/dlya-devochek/odezhda/verhnyaya-odezhda 100
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-devochek/vodolazki 30
https://www.wildberries.ru/catalog/detyam/odezhda/dlya-devochek/vodolazki 100


'''

init_urls = []

for line in urls_config.splitlines():
    line = line.strip()
    if line.startswith("#"): continue
    if not line.startswith("https://"): continue
    try:
        url, page_count = line.split()
        page_count = min(int(page_count),10)
        if not url.endswith('/'): url = url + '/'
        for i in range(1,page_count):
            init_urls.append(url+'?sort=popular&page='+str(i))
    except Exception as err:
        logging.warning(f"error parsing line {line}: {err}")


logging.info(f'Total links = {len(init_urls)}')
logging.info(f'{init_urls[:5]}')
Crawler.start_crawling(init_urls)


    



