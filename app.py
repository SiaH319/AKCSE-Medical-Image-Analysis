from bs4 import BeautifulSoup
import requests


source = requests.get('http://127.0.0.1:5502/scraping.html').text

soup = BeautifulSoup(source, 'lxml')

print(soup.prettify())




@app.route ('/post', methods= ['POST'])
def post():
    value = request.form ['test']
    return value



