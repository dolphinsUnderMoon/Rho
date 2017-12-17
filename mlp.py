from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import h5py


inputs = Input(shape=(3,))

x = Dense(128, activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

train_x = np.load("./data/train_x.npy")
train_y = np.load("./data/train_y.npy")
test_x = np.load("./data/test_x.npy")
test_y = np.load("./data/test_y.npy")

model.fit(train_x, train_y, batch_size=16, epochs=20)

model.save_weights("./model/model.h5")
model_structure = model.to_json()
with open("./model/model.json", 'w') as f:
    f.write(model_structure)

score = model.evaluate(test_x, test_y)
print(score)