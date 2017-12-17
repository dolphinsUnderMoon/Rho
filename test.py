from keras.models import Model
import numpy as np
from keras.models import model_from_json
import h5py

test_x = np.load("./data/test_x.npy")
test_y = np.load("./data/test_y.npy")

with open("./model/model.json", 'r') as f:
    model_structure = f.read()

model = model_from_json(model_structure)
model.load_weights("./model/model.h5")

# choose an index randomly
sample_index = 24
sample = test_x[sample_index].reshape([1, 3])
sample_target_value = test_y[sample_index]

result = model.predict(sample)
print(result, sample_target_value)