from PIL import Image
import glob
import os

def formatIm(currDir):
    c=1
    for filename in glob.glob(currDir + 'dataset-raw/*.*'):
        im=Image.open(filename)
        folder = 'dataset-mask/img'+str(c)+'/'
        os.makedirs(currDir + folder)
        name='img'+str(c)+'.png'
        format_im = im.convert('RGBA')
        format_im = im.resize((512,512), Image.ANTIALIAS)
        format_im.save(currDir + 'dataset-formatted/' + name)
        format_im.save(currDir + folder + name)
        c+=1 
    print ("images resized and reformatted")

formatIm('c:/Users/hamsi/OneDrive/Desktop/AKCSE-Medical-Image-Analysis/')