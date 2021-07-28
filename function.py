import numpy as np
import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image


def image_float_to_uint(image):
    if np.max(image) <= 1:  # image between [0, 1]
        return (image * 255).astype(np.uint8)
    else:  # image between [0, 255]
        return image.astype(np.uint8)


def image_uint_to_float(image):
    if np.max(image) <= 1:  # image between [0, 1]
        return image
    else:
        return (image / 255).astype(np.float)


def get_all_img(directory):
    images = sorted(os.listdir(os.path.join(directory)))  # os.listdir gives a list of all files name in this path
    data = []
    for image in images:
        data.append(mpimg.imread(os.path.join(directory, image)))
    return np.array(data)


def plot_before_after(before, after, file_name=""):
    plt.figure(figsize=(15, 5))  # Size of figure (all plot)

    plt.subplot(1, 2, 1)  # (nrows, ncols, index)
    plt.axis('off')
    plt.title("Before", fontsize=20)
    plt.imshow(before)

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title("After", fontsize=20)
    plt.imshow(after)

    if not file_name == "":
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def img_resize(img_array, new_size):
    # Convert Image to array with np.array(Image) & convert np.array to Image with Image.fromarray(array)
    return np.array(Image.fromarray(img_array).resize(new_size))


def resize_all_img(directory, new_shape):
    # Creat new folder
    folder_name = directory[- directory[::-1].index('\\'):]
    new_folder_name = folder_name + str(new_shape)
    new_path = os.path.join(directory[:-len(folder_name)], new_folder_name)
    Path(new_path).mkdir(parents=True, exist_ok=True)

    # Resize images and save new image
    images = sorted(os.listdir(os.path.join(directory)))  # os.listdir gives a list of all files name in this path
    for name in images:
        image = Image.open(os.path.join(directory, name))
        image = image.resize(new_shape)
        image.save(os.path.join(new_path, str(name)))

    print("Image resizing completed")
