# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:07:54 2017

@author: sparsh
"""

"""
Crop Disease Classification Project for Code Fun Do 2017 - IIT Roorkee
"""

"""
File for downloading the dataset using URLs in CSV from the PlantVillage 
dataset.
"""

import pandas as pd
import requests
import shutil

datasetPath = "E:\\Interesting\\Code Fun Do 2017\\corn_maize_images.csv"
datasetFrame = pd.read_csv(datasetPath, header=0)
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\cercospora_leaf_spot_"

#path = datasetFrame['url'][1]
#r = requests.get(path, stream=True)
#if r.status_code == 200:
#    with open(imagePathPref, 'wb') as f:
#        r.raw.decode_content = True
#        shutil.copyfileobj(r.raw,f)
"""
Cercospora leaf spot 0 - 512
for i in range(419,513):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
"""
"""
#Healthy 513-1674
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\healthy_"
for i in range(513,1675):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i-513) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
print("Healthy Batch 1 complete.\n")

#Common rust 1675-2866
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\common_rust_"
for i in range(1675,2867):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i-1675) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
print("Common rust Batch 1 complete.\n")

#Northern Leaf Blight 2867-3851
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\northern_leaf_blight_"
for i in range(3713,3852):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i-2867) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
print("Northern Leaf Blight Batch 1 complete.\n")

#Cercospora leaf spot 3852-4795
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\cercospora_leaf_spot_"
for i in range(3852,4796):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i-3852+513) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
print("Common leaf spot Batch 2 complete.\n")

#Common rust 4796-5217
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\common_rust_"
for i in range(4796,5218):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i-4796+1192) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
print("Common rust Batch 2 complete.\n")
"""
#Healthy 5218-8505
imagePathPref = "E:\\Interesting\\Code Fun Do 2017\\Dataset\\healthy_"
for i in range(8211,8506):
    img_url = datasetFrame['url'][i]
    imagePath = imagePathPref + str(i-5218+1162) + ".jpg"
    r = requests.get(img_url, stream = True)
    if r.status_code == 200:
        with open(imagePath, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw,f)
print("Healthy Batch 2 complete.\n")

print("Download complete.\n")
#print(datasetFrame['url'][0])