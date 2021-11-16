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

from requests.models import parse_url

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# proxies
# https://www.scrapehero.com/how-to-rotate-proxies-and-ip-addresses-using-python-3/


headers = {
    # 'authority': 'www.ozon.ru',
    'cache-control': 'max-age=0',
    'upgrade-insecure-requests': '1',
    # 'user-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36'
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36',
    'sec-fetch-dest': 'document',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
}

session = requests.session()


def get_proxies():
    proxy_url_list = 'https://free-proxy-list.net/'
    response = requests.get(proxy_url_list)
    soup = BeautifulSoup(response.text, 'html.parser')
    # textarea class="form-control"
    ips_list = soup.find("textarea.form-control")
    https_proxy_list = []
    return https_proxy_list

class Crawler:
    DATA_DIR = Path('/home/deeplab/datasets/custom_fashion/data/lamoda_ru')

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
        session =  requests.session()
        return session

 
    def parse_page(self, url):
        RESOLUTION_PATH = {'300':'img236x341','500': 'img389x562','800': 'img600x866','full':'product'}
        img_height_option = RESOLUTION_PATH['800']

        session =self._get_session()
        time.sleep(1)
        response = session.get(url, headers=headers)
        if response.status_code !=200:
            logging.warning(f'Bad code={response.status_code} for {url}')
        soup = BeautifulSoup(response.text, 'html.parser')

        html_garment_list = soup.find_all("div",{"class":"products-list-item"})
        for index,item in enumerate(html_garment_list):
            try:
                meta = {}
                id = item.select_one("[data-sku]").get("data-sku")
                gallery_list = json.loads(item['data-gallery'])
                meta['gallery_list'] = gallery_list
                try:
                    meta['name'] = item.select_one("[data-name]").get("data-name")
                except Exception: logging.warning(f' no name: id={id} , url={url} , index={index}')
                try:
                    meta['category'] = item.select_one("[data-category]").get("data-category")
                except Exception: logging.warning(f' no category: id={id} , url={url} , index={index}')
                try:
                    meta['gender'] = item.select_one("[data-gender]").get("data-gender")
                except Exception: logging.warning(f' no gender: id={id} , url={url} , index={index}')
                try:
                    meta['brand'] = item.select_one("[data-brand]").get("data-brand")
                except Exception: logging.warning(f' no brand: id={id} , url={url} , index={index}')
                try:
                    meta['color'] = item.select_one("[data-color-family]").get("data-color-family")
                except Exception: logging.warning(f' no color: id={id} , url={url} , index={index}')
                
                meta['id']=id
                # https://a.lmcdn.ru/img600x866/L/E/LE306EWFOWD5_8853255_1_v2_2x.jpg

                headers['referer'] = url
                garment_dir = Path(self.DATA_DIR, id)
                garment_dir.mkdir(parents=True,exist_ok=True)

                for img_link in gallery_list:
                    # replace //a.lmcdn.ru/img236x341/V/I/VI004EWLVIA8_13046308_1_v1.jpg to
                    # https://a.lmcdn.ru/img600x866/L/E/LE306EWFOWD5_8853255_1_v2_2x.jpg
                    link_800 = re.sub(r"^(//)(.+?)/(.+?)/(.+)", r'https://\2/'+RESOLUTION_PATH['800']+r'/\4', img_link)

                    
                    image_filename = Path(garment_dir , link_800[link_800.rindex('/')+1:])                
                    if not image_filename.exists():
                        time.sleep(0.3)
                        img_resp = session.get(link_800, headers=headers, stream=True)                
                        if img_resp.status_code == 200:
                            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                            img_resp.raw.decode_content = True
                            # Open a local file with wb ( write binary ) permission.
                            with open(image_filename,'wb') as f:
                                shutil.copyfileobj(img_resp.raw, f)
                        else:
                            logging.warning(f'Image dowload bad code={img_resp.status_code} for {link_800}')
                    # else:
                    #     logging.warning(f'File exitst size={image_filename.stat().st_size} file={image_filename} for {link_800}')
                        
                json_file = Path(garment_dir , "meta.json")
                with open(json_file, 'w') as fp:
                    json_string = json.dumps(meta, default=lambda o: o.__dict__, sort_keys=True, indent=2, ensure_ascii=False)
                    fp.write(json_string)
            except:
                logging.warning(f' error parsing block: url={url} , index={index}')
        
        self._set_parsed_page(url)





urls_config = '''
# man
https://www.lamoda.ru/c/517/clothes-muzhskie-bryuki/ 100
https://www.lamoda.ru/c/479/clothes-muzhskaya-verkhnyaya-odezhda 120
https://www.lamoda.ru/c/497/clothes-muzhskoy-trikotazh 60
https://www.lamoda.ru/c/513/clothes-muzhskie-d-insy 50
https://www.lamoda.ru/c/5289/clothes-odezhdadlyadomamuj 15
https://www.lamoda.ru/c/7660/clothes-men-kombenizony 1
https://www.lamoda.ru/c/3039/clothes-topyi-muzhskie 8
https://www.lamoda.ru/c/7528/clothes-bigsize-clothes-men/ 167
https://www.lamoda.ru/c/3043/clothes-pid-aki-kostumi-muzhskie/ 40
https://www.lamoda.ru/c/2526/clothes-plyazhnaya-odezhda/ 18
https://www.lamoda.ru/c/515/clothes-muzhskie-rubashki-i-sorochki/ 100
https://www.lamoda.ru/c/3042/clothes-sportivnye-kostyumy-muzhskie/ 20
https://www.lamoda.ru/c/5295/clothes-termobelyemuj/ 2
https://www.lamoda.ru/c/2508/clothes-tolstovki-i-olimpiyki/ 110
https://www.lamoda.ru/c/2512/clothes-muzhskie-futbolki/ 167
https://www.lamoda.ru/c/519/clothes-muzhskie-shorty/ 50



# womens
https://www.lamoda.ru/c/399/clothes-bluzy-rubashki/ 167
https://www.lamoda.ru/c/4418/clothes-body/ 15
https://www.lamoda.ru/c/401/clothes-bryuki-shorty-kombinezony/ 167
https://www.lamoda.ru/c/357/clothes-verkhnyaya-odezhda 167
https://www.lamoda.ru/c/7571/default-knitted-suits 10
https://www.lamoda.ru/c/371/clothes-trikotazh 167
https://www.lamoda.ru/c/397/clothes-d-insy 110
https://www.lamoda.ru/c/4651/clothes-dom-odejda/ 110
https://www.lamoda.ru/c/4184/clothes-coveralls/ 34
https://www.lamoda.ru/c/3002/clothes-plyajnaya-odejda 100
https://www.lamoda.ru/c/709/clothes-nizhneye-belyo 167
https://www.lamoda.ru/c/2937/clothes-clothes-big-size/ 167
https://www.lamoda.ru/c/4170/clothes-maternityclothes 40
https://www.lamoda.ru/c/367/clothes-pidzhaki-zhaketi/ 100
https://www.lamoda.ru/c/415/clothes-kostyumy/ 80
https://www.lamoda.ru/c/2474/clothes-tolstovki-olimpiyki 150
https://www.lamoda.ru/c/2627/clothes-topy 120
https://www.lamoda.ru/c/4748/clothes-womtuniki/ 5
https://www.lamoda.ru/c/2480/clothes-futbolki-s-dlinnym-rukavom 30
https://www.lamoda.ru/c/2478/clothes-futbolki 167
https://www.lamoda.ru/c/2485/clothes-shorty 80
https://www.lamoda.ru/c/423/clothes-yubki/ 140

# child
'''

init_urls = []

for line in urls_config.splitlines():
    line = line.strip()
    if line.startswith("#"): continue
    if not line.startswith("https://"): continue
    try:
        url, page_count = line.split()
        page_count = int(page_count)
        if not url.endswith('/'): url = url + '/'
        for i in range(1,page_count):
            init_urls.append(url+'?page='+str(i))
    except Exception as err:
        logging.warning(f"error parsing line {line}: {err}")


logging.info(f'Total links = {len(init_urls)}')
logging.info(f'{init_urls[:5]}')
Crawler.start_crawling(init_urls)



    



