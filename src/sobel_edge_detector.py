from PIL import Image
import numpy as np
from math import sqrt

min_edgyness = 600

HK = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]

VK = [[1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]]

def array_from_image(load_filepath):
    """Load image and convert to 2D pixel array"""
    im = Image.open(load_filepath)
    im_pixels = list(im.getdata())
    width, height = im.size

    # Convert the image to a 2D array of pixel values
    im_pixels = [im_pixels[i * width:(i + 1) * width] for i in range(height)]
    return im_pixels

def image_from_array(pixel_array, save_filepath):
    """Convert 2D pixel array back to image and save"""
    array = np.array(pixel_array, dtype=np.uint8)
    new_image = Image.fromarray(array)
    new_image.save(save_filepath)

def image_to_gray(pixel_array):
    """Convert image to grayscale"""
    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[i])):
            pixel = pixel_array[i][j]
            av_color = (pixel[0] + pixel[1] + pixel[2]) // 3
            pixel_array[i][j] = (av_color, av_color, av_color)
            
    return pixel_array

def getvalue(array, x, y):
    """Get pixel value, returns 0 if on edge"""
    try:
        return array[x][y][0]
    except IndexError:
        return 0

def get3X3matrix(array, x, y):
    """Get 3x3 pixel matrix around position (x, y)"""
    pixelmatrix = [[getvalue(array, x - 1, y - 1), getvalue(array, x - 1, y), getvalue(array, x - 1, y + 1)],
                   [getvalue(array, x, y - 1), getvalue(array, x, y), getvalue(array, x, y + 1)],
                   [getvalue(array, x + 1, y - 1), getvalue(array, x + 1, y), getvalue(array, x + 1, y + 1)]]
    return pixelmatrix

def getedgyness(PM):
    """Calculate edge intensity using Sobel kernels"""
    HKV = 0
    VKV = 0
    
    for n in range(len(PM)):
        for m in range(len(PM[0])):
            HKV += HK[n][m] * PM[n][m]
            VKV += VK[n][m] * PM[n][m]
    
    KV = sqrt(HKV * HKV + VKV * VKV)
    return KV

def image_to_blur(pixel_array):
    """Apply blur filter to image"""
    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[i])):
            PM = get3X3matrix(pixel_array, i, j)
            
            Irow1 = PM[0][0] + PM[0][1] + PM[0][2]
            Irow2 = PM[1][0] + PM[1][1] + PM[1][2]
            Irow3 = PM[2][0] + PM[2][1] + PM[2][2]
            I = (Irow1 + Irow2 + Irow3) / 9
            
            pixel_array[i][j] = (I, I, I)
            
    return pixel_array

def detect_edges_sobel(input_path, output_path):
    """Detect edges using Sobel operator"""
    # Load the image and convert it to a 2D pixel array
    pixels = array_from_image(input_path)
    image_from_array(pixels, "output/step1_original.png")

    # Convert to grayscale
    pixels = image_to_gray(pixels)
    image_from_array(pixels, "output/step2_grayscale.png")
    
    # Apply blur
    pixels = image_to_blur(pixels)
    image_from_array(pixels, "output/step3_blurred.png")

    # Apply Sobel operator
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            PM = get3X3matrix(pixels, i, j)
            edgyness = getedgyness(PM)
            pixels[i][j] = (edgyness, edgyness, edgyness)
            
    # Save the processed image
    image_from_array(pixels, output_path)

if __name__ == "__main__":
    detect_edges_sobel("samples/sample_image.jpg", "output/sobel_edges.png")
