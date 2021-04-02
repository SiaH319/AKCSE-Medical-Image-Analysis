from bs4 import BeautifulSoup
import requests

source = requests.get('http://127.0.0.1:5502/index-project.html').text

soup = BeautifulSoup(source, 'lxml')

print(soup.prettify())


print(soup)

article = soup.find('div', class_='container')
# print(article)

headline = container.h2
print(headline)
