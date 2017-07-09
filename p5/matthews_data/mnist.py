import os
import keras
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from random import choice, random
from string import ascii_lowercase, digits
alphanumeric = ascii_lowercase + digits
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
width, height, n_len, n_class = 100, 50, 4, len(alphanumeric)

batch_size = 32
epochs = 120
input_shape = (height, height, 1)


# input image dimensions
img_rows, img_cols = 50, 50

def fuzzy_loc(locs):
    acc = []
    for i,loc in enumerate(locs[:-1]):
        if locs[i+1] - loc < 8:
            continue
        else:
            acc.append(loc)
    return acc

def seg(img):
    arr = np.array(img, dtype=np.float)
    arr = arr.transpose()
    # arr = np.mean(arr, axis=2)
    arr = np.sum(arr, axis=1)
    locs = np.where(arr < arr.min() + 2)[0].tolist()
    locs = fuzzy_loc(locs)
    return locs

def is_well_formed(img_path):
    original_img = Image.open(img_path)
    img = original_img.convert('1')

    return len(seg(img)) == 4

noiseimg = np.array(Image.open("Average.png").convert("1"))
noiseimg = np.bitwise_not(noiseimg)
fnt = ImageFont.truetype('../arial-extra.otf', 26)
def gen_one():
    og = Image.new("1", (100,50))
    text = ''.join([choice(alphanumeric) for _ in range(4)])
    draw = ImageDraw.Draw(og)
    for i, t in enumerate(text):
        txt=Image.new('L', (40,40))
        d = ImageDraw.Draw(txt)
        d.text( (0, 0), t,  font=fnt, fill=255)
        if random() > 0.5:
            w=txt.rotate(-40*(random()-1),  expand=1)
            og.paste( w, (i*int(20) + int(25*random()), int(25+40*(random()-1))),  w)
        else:
            w=txt.rotate(40*(random()-1),  expand=1)
            og.paste( w, (i*int(20) + int(25*random()), int(30*random())),  w)

    segments = seg(og)
    if len(segments) != 4:
        return gen_one()
    ogarr = np.array(og)
    ogarr = np.bitwise_and(noiseimg, ogarr)
    ogarr = np.expand_dims(ogarr, axis=2).astype(float)
    ogarr = np.random.random(size=(50,100,1)) * ogarr
    ogarr = (ogarr > 0.1).astype(float)
    return ogarr, text, segments


def pad(a, shape):
    result = np.zeros(shape)
    a = a[:shape[0], :shape[1]]
    result[:a.shape[0],:a.shape[1]] = a
    return result

def gen_segmented(arr, segments):
    sqz_arr = np.squeeze(arr,2)
    char1 = sqz_arr[:, segments[0]:segments[1]]
    char2 = sqz_arr[:, segments[1]:segments[2]]
    char3 = sqz_arr[:, segments[2]:segments[3]]
    char4 = sqz_arr[:, segments[3]:]
    char1x = pad(char1, (50,50))
    char2x = pad(char2, (50,50))
    char3x = pad(char3, (50,50))
    char4x = pad(char4, (50,50))
    return np.array([char1x, char2x, char3x, char4x])

def gen_char_train():
    arr, txt, segments = gen_one()
    return gen_segmented(arr, segments), txt
def gen_char_infer():
    path = choice(img_paths)
    if not is_well_formed(path):
        return gen_char_infer()
    randimg = Image.open(path).convert('1')
    segments = seg(randimg)
    imarr = np.array(randimg).astype(float)
    imarr = np.expand_dims(imarr, axis = 2)
    return gen_segmented(imarr, segments)


def gen_char(batch_size=batch_size):
    X = np.zeros((batch_size * 4, height, height), dtype=np.uint8)
    y = np.zeros((batch_size * 4))
    while True:
        for i in range(batch_size):
            im, text = gen_char_train()
            for j, ch in enumerate(text):
                X[i*4 + j] = im[j]
                # y[j][i, :] = 0
                # y[j][i, alphanumeric.find(ch)] = 1
                y[i*4 + j] = alphanumeric.find(ch)
        yield X, np.array(y)


def gen_data(batch_size):
  x, y = gen_char(batch_size).next()
  # x = np.concatenate(x, axis=0)
  x = np.expand_dims(x, 3)
  return x, y


# x, y = gen_data(6)
# print x.shape
# print y.shape

def load_data():
    X = []
    Y = []
    with open("solved.txt") as f:
        images = [i.split(',') for i in f.read().split('\n')]
    for image in images:
        name = image[0]
        answer = image[1]
        path = "./orig/{}.jpg".format(name)
        if is_well_formed(path):
            randimg = Image.open(path).convert('1')
            segments = seg(randimg)
            imarr = np.array(randimg).astype(float)
            imarr = np.expand_dims(imarr, axis = 2)
            splitted_chars_arr = gen_segmented(imarr, segments)
            X.append(splitted_chars_arr)
            for char in answer:
                Y.append(alphanumeric.find(char))
    X = np.concatenate(X, axis=0)
    X = np.expand_dims(X, 3)

    num_samples = len(X)
    stop = int(num_samples * 0.95)
    return (np.array(X[:stop]), np.array(Y[:stop])), (np.array(X[stop:]), np.array(Y[stop:]))

# (x_train, y_train), (x_test, y_test) = load_data()

# y_train = keras.utils.to_categorical(y_train, n_class)
# y_test = keras.utils.to_categorical(y_test, n_class)

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
for _ in range(4):
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# model.save("mnist.h5")
model.load_weights("mnist.h5")

# raise Exception

def decode(y):
    y = np.argmax(np.array(y), axis=1)
    return ''.join([alphanumeric[x] for x in y])


jpg_dir = './orig'
img_paths = [jpg_dir + '/' + i for i in os.listdir(jpg_dir)]
import matplotlib.pyplot as plt

def infer_img():
    X = gen_char_infer()
    X = np.expand_dims(X, 3)
    y_pred = model.predict(X)
#     print np.max(y_pred[0],axis=1)
    plt.title('real: ???\npred:%s'%(decode(y_pred)))
    print X.shape
    plt.imshow(np.concatenate(np.squeeze(X, 3), 1), cmap='gray')
    plt.axis('off')
    plt.show()
    return decode(y_pred)

# print infer_img()


# def predict_production(img_path):
#     randimg = Image.open(img_path).convert('1')
#     segments = seg(randimg)
#     imarr = np.array(randimg).astype(float)

#     imarr = np.expand_dims(imarr, axis = 2)
#     X = gen_segmented(imarr, segments)
#     X = np.expand_dims(X, 3)
#     y_pred = model.predict(X)
#     return img_path[7:-4], decode(y_pred)

# with open("outfile2", 'w') as f:
#     for i, p in enumerate(img_paths):
#         if i % 5000 == 0:
#             print i
#         if is_well_formed(p):
#             p, pred = predict_production(p)
#             f.write(",".join([p, pred]))
#             if i != len(img_paths):
#                 f.write("\n")




import requests, json
from threading import Thread

url = "https://captcha.delorean.codes/u/MatthewMerrill/solution"

def post(list_of_solved):
    r = requests.post(url, data=json.dumps({"solutions": list_of_solved}))
    print r.status_code
    print r.text

data = []
ts = []
solutions = [i.split(',') for i in open('outfile2', 'r').read().split('\n')]
print len(solutions)
for i in range(len(solutions) - 15000):
    for sol in solutions[i:15000+i]:
        data.append({'name': sol[0], 'solution': sol[1]})

    print len(data)
    post(data)
    t = Thread(target=post, args=(data,))
    t.start()
    data = []