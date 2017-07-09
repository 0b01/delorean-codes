import keras.backend as K
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from PIL import Image
import numpy as np
import tensorflow as tf
from pprint import pprint

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    img = img.resize((32,32))
    data = np.asarray( img, dtype="int32" )
    return data


model_path = "model.json"
weights_path = 'model.hdf5'

with open(model_path) as f:
    txt = f.read()
model = model_from_json(txt)
model.load_weights(weights_path)

layer_dict = dict([(layer.name, layer) for layer in model.layers])
pprint(layer_dict)

# print(model.summary())
# inp = load_image('car.jpg')
# inp = np.expand_dims(inp, axis=0)
# print model.predict(x=inp, batch_size=1)

input_img = model.input

output_index = 1
batch_size = 1
loss = K.mean(model.outputs[0][:, output_index])

grads = K.gradients(loss, input_img)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([input_img], [loss, grads])

input_img_data = np.random.random((1, 32, 32, 3)) * 20 + 128.

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

step = 1
for i in range(1000):
    print i
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

print model.predict(x=input_img_data, batch_size=1)
img = deprocess_image(input_img_data[0])

im = Image.fromarray(img)
im.save('1.png')


# https://captcha.delorean.codes/u/rickyhan/challenge