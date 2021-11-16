import requests
from bs4 import BeautifulSoup


# proxies
# https://www.scrapehero.com/how-to-rotate-proxies-and-ip-addresses-using-python-3/
class Prodcut:
    name = str
    price = str
    link = str

    def __init__(self, name, price, link):
        self.name = name
        self.price = price
        self.link = link

    def __repr__(self):
        return str(self.__dict__)

headers = {
    'authority': 'www.ozon.ru',
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

response = session.get('https://www.ozon.ru/category/povsednevnye-platya-zhenskie-36405/?page=14', headers=headers)

soup = BeautifulSoup(response.text, 'html.parser')
# soup.find_all("div", {"class": "a0c6"})
for element in soup.find_all('div', class_='a0c6'):
    name = element.find('h1', class_="product-card__title").text.strip()
    price = element.find('span', class_="product-card__price").text.strip()
    link = "https://kith.com/" + element.find('a').get('href')

    prodcut = Prodcut(name, price, link)

    print(prodcut.__repr__())