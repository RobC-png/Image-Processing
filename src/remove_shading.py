import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def remove_shading_and_keep_colors(image_path, output_path, num_colors=8, brightness_threshold=150):
    """
    Remove shading from image while preserving colors using K-means segmentation.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        num_colors: Number of color clusters for segmentation
        brightness_threshold: Threshold for keeping bright areas
    """
    try:
        # Suppress the physical cores warning
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'

        # Read the image
        image = cv2.imread(image_path)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to keep bright areas
        _, mask = cv2.threshold(gray_image, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

        # Reshape the image to a 2D array of pixels
        pixels = masked_image.reshape((-1, 3))

        # Perform k-means clustering for color quantization
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)

        # Assign each pixel to the nearest cluster center
        segmented_image = kmeans.cluster_centers_[kmeans.labels_]

        # Reshape the segmented image back to the original shape
        segmented_image = segmented_image.reshape(image_rgb.shape)

        # Convert the image back to BGR
        segmented_image_bgr = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Save the segmented image
        cv2.imwrite(output_path, segmented_image_bgr)

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")

if __name__ == "__main__":
    remove_shading_and_keep_colors('samples/sample_image.jpg', 'output/no_shading.jpg', 
                                   num_colors=8, brightness_threshold=150)
