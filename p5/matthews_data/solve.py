import json
import base64
import os

def save_imgs_matthew_customdata(fname):
	with open(fname) as f:
		l = [i.split(',') for i in f.read().split('\n')]

		for image in l:
			b = base64.decodestring(image[2])
			name = image[0]
			with open("./orig/{}.jpg".format(name), 'w') as f:
				f.write(b)
			print name



def save_imgs(fname):
    with open(fname) as f:
        l = json.loads(f.read())

    for image in l['images']:
        b = base64.decodestring(image['jpg_base64'])
        name = image['name']
        with open("./orig/{}.jpg".format(name), 'w') as f:
            f.write(b)
        print name

challenges_dir = './challenges'
for p in os.listdir(challenges_dir):
    save_imgs(challenges_dir + '/' + p)

save_imgs('solved.txt')
