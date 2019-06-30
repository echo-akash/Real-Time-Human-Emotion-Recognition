import pandas as pd

raw_data_csv_file_name = 'data/fer2013.csv'
raw_data = pd.read_csv(raw_data_csv_file_name)

raw_data.info()

raw_data.head()

raw_data["Usage"].value_counts()

import matplotlib
import matplotlib.pyplot as plt

def show_image_and_label(x, y):
    x_reshaped = x.reshape(48,48)
    plt.imshow(x_reshaped, cmap= "gray",
              interpolation="nearest")
    plt.axis("off")
    plt.show()
    print(y)

import numpy as np

img = raw_data["pixels"][0]
val = img.split(" ")
x_pixels = np.array(val, 'float32')
x_pixels /= 255

show_image_and_label(x_pixels, raw_data["emotion"][0])
