import os
import json
import requests  # to sent GET requests
from bs4 import BeautifulSoup


# user can input a topic and a number
# download first n images from google image search

GOOGLE_IMAGE = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

# The User-Agent request header contains a characteristic string
# that allows the network protocol peers to identify the application type,
# operating system, and software version of the requesting software user agent.
# needed for google search
usr_agent = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

SAVE_FOLDER = 'imagesFound'


def main():
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    download_images()


def download_images():
    # ask for user input
    data = input('What are you looking for? ')
    n_images = int(input('How many images do you want? '))

    print('Start searching...')

    # get url query string
    searchurl = GOOGLE_IMAGE + 'q=' + data
    print(searchurl)

    # request url, without usr_agent the permission gets denied
    response = requests.get(searchurl, headers=usr_agent)
    html = response.text
    print("html: ", html)

    # find all divs where class='rg_meta'
    soup = BeautifulSoup(html, 'html.parser')
    print("soup: ", soup)
    #print(soup)
    #results = soup.findAll('div', {'class': '<img class="rg_i'}, limit=n_images)
    #above comment doesn't work anymore since google changed html sturcture

    #results = soup.findAll('div', {'class': 'RAyV4b'}, limit=n_images)
    results = soup.findAll('img', {'alt': '', 'class': 't0fcAb'}, limit=n_images)
    print("results:", results)

    # gathering requested number of list of image links with data-src attribute
    # continue the loop in case query fails for non-data-src attributes
    count = 0
    links = []
    for res in results:
        try:
            link = res['src']
            links.append(link)
            count += 1
            if (count >= n_images): break

        except KeyError:
            continue
    print ("count: ", count)
    print("links: ", links)

    print(f'Downloading {len(links)} images....')

    # Access the data URI and download the image to a file
    for i, link in enumerate(links):
        response = requests.get(link)

        image_name = SAVE_FOLDER + '/' + data + str(i + 1) + '.png'
        with open(image_name, 'wb') as fh:
            fh.write(response.content)



    """

    # extract the link from the div tag
    imagelinks = []
    for result in results:
        text = result.text  # this is a valid json string
        #print(text)

        text_dict = json.loads(text)  # deserialize json to a Python dict
        link = text_dict['ou']
        # image_type = text_dict['ity']
        imagelinks.append(link)

    print(f'found {len(imagelinks)} images')
    print('Start downloading...')

    for i, imagelink in enumerate(imagelinks):
        # open image link and save as file
        response = requests.get(imagelink)

        imagename = SAVE_FOLDER + '/' + data + str(i + 1) + '.jpg'
        with open(imagename, 'wb') as file:
            file.write(response.content)

    print('Done')
    """



if __name__ == '__main__':
    main()
