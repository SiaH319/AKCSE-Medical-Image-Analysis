from bs4 import BeautifulSoup
import requests

source = requests.get('http://127.0.0.1:5502/index-main.html').text

soup = BeautifulSoup(source, 'lxml')


print(soup.prettify())




