import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from PIL import Image, ImageFont, ImageDraw

images = [
    "BlueNoise16", "BlueNoise64",
    "perlin_2", "perlin_4", "perlin_8", "perlin_16", "perlin_32", "perlin_64", "perlin_64_smooth", "perlin_64_none", "perlin_128",
    "perlin_white_same_1", "perlin_white_same_2", "perlin_white_same_3",
    "perlin_white_different_1", "perlin_white_different_2", "perlin_white_different_3",
    "perlin_blue_same_1", "perlin_blue_same_2", "perlin_blue_same_3",
    "perlin_blue_different_1", "perlin_blue_different_2", "perlin_blue_different_3",
    "perlin_ign_same_1", "perlin_ign_same_2", "perlin_ign_same_3",
    "perlin_ign_different_1", "perlin_ign_different_2", "perlin_ign_different_3",
    "perlin_big_blue64x64", "perlin_big_blue16x16", "perlin_big_white64x64", "perlin_big_white16x16"
]

for image in images:
    img_c1 = imageio.imread(image + ".png")

    if img_c1.ndim == 3:
        img_c1 = np.dot(img_c1[... , :3] , [0.21 , 0.72, 0.07]) 

    img_c2 = np.fft.fft2(img_c1)
    img_c2 = np.fft.fftshift(img_c2)
    img_c2 = np.log(1+np.abs(img_c2))
    imageio.imwrite(image + ".mag.png", img_c2)
