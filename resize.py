import os
import os.path
from PIL import Image

f = r'D://AB-Farrel/img'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((100,100))
    img.save(f_img)