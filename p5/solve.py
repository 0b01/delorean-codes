import json
import pytesseract
import numpy as np
from pprint import pprint
import base64
import os
from PIL import Image, ImageFilter, ImageChops, ImageDraw
import mahotas
import cv2
import matplotlib.pyplot as plt

from pylab import imshow, show

def save_imgs(fname):
    with open(fname) as f:
        l = json.loads(f.read())

    for image in l['images']:
        b = base64.decodestring(image['jpg_base64'])
        name = image['name']
        with open("./orig/{}.jpg".format(name), 'w') as f:
            f.write(b)
        print name

# challenges_dir = './challenges'
# for p in os.listdir(challenges_dir):
#     save_imgs(challenges_dir + '/' + p)

def gray(img_path):

    img = Image.open(img_path).convert("L")

    img = img.point(lambda x: 255 if x > 200 or x == 0 else x)
    img = img.point(lambda x: 0 if x < 255 else 255)

    img.save(img_path)

jpg_dir = './matthews_data/orig'
for i, p in enumerate(os.listdir(jpg_dir)):
    print i
    img_path = jpg_dir + '/' + p
    gray(img_path)
print 'done'


def avg():
    jpg_dir = './matthews_data/orig'
    img_paths = [jpg_dir + '/' + i for i in os.listdir(jpg_dir)]


    w,h=Image.open(img_paths[0]).size
    N=len(img_paths)
    arr=np.zeros((h,w),np.float)

    for im in img_paths:
        imarr=np.array(Image.open(im),dtype=np.float)
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="L")
    out.save("Average_matt.png")

avg()

im = Image.open("Average_matt.png")
im = im.point(lambda x:255 if x > 230 else x)
im = im.point(lambda x:0 if x<255 else 255)
im.save("Average_matt.png")

def deviate():
    jpg_dir = './matthews_data/orig'
    img_paths = [jpg_dir + '/' + i for i in os.listdir(jpg_dir)]
    average = Image.open("./matthews_data/Average.png")
    point_table = lambda x: 255 if x > 55 else 0

    for im in img_paths:
        img = Image.open(im)
        diff = ImageChops.difference(average, img)

        # diff = diff.convert('L')
        # diff = diff.convert('1')
        # new.paste(img, mask=diff)

        # diff.show()
        diff.save(im)

deviate()

def fuzzy_loc(locs):
    acc = []
    for i,loc in enumerate(locs[:-1]):
        if locs[i+1] - loc < 8:
            continue
        else:
            acc.append(loc)
    return acc

def to_text(im):
    text = pytesseract.image_to_string(im)
    return text
def seg():
    jpg_dir = './jpgs'
    img_paths = [jpg_dir + '/' + i for i in os.listdir(jpg_dir)][6:]
    for im in img_paths:
        original_img = Image.open(im)

        w,h = original_img.size
        img = original_img.convert('1')
        arr = np.array(img, dtype=np.float)
        arr = arr.transpose()
        # arr = np.mean(arr, axis=2)
        arr = np.sum(arr, axis=1)
        locs = np.where(arr < arr.min() + 2)[0].tolist()
        locs = fuzzy_loc(locs)

        # for l in locs:
        #     plt.axvline(x=l)
        # plt.plot(np.arange(0,100), arr)
        # plt.show()
        locs += [100]
        locs = [0] + locs

        draw = ImageDraw.Draw(original_img)
        for i in range(len(locs) - 1):
            subimg = original_img.crop((locs[i], 0, locs[i+1], h))
            print to_text(subimg)
            # subimg.show()
            draw.line((locs[i],0,locs[i],h), fill=255)
        original_img.show()
        original_img.save("SAVED.png")
        break

# seg()