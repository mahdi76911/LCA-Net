from LCA_Net import *
from function import *

from os.path import join

#######################################################################################################################
# Load Dataset & Preprocessing

org_image_shape = (460, 620)
image_shape = (512, 512)

dataset_path = './Dataset/'
save_filename = './Results/result01.png'

# X & y are Normalized ([0, 1])
X = get_all_img(join(dataset_path, 'Haze(512, 512)'))
y = get_all_img(join(dataset_path, 'Clear(512, 512)'))

#######################################################################################################################
# Build model

# Build LCA_Net Model
model = LCA_Net(image_shape)

model.fit(X, y, batch_size=16, epochs=5, verbose=1)
