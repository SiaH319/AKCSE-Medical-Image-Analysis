from bs4 import BeautifulSoup
import requests

with open('index-project.html') as html_file:
    soup = BeautifulSoup(html_file, 'lxml')

print(soup)

article = soup.find('div', class_='container')
# print(article)

headline = container.h2
print(headline)
