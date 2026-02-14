import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def extract_bright_colorful_colors(image_path, num_colors=8, brightness_threshold=150, saturation_threshold=50):
    """
    Extract bright and colorful colors from an image using K-means clustering.
    
    Args:
        image_path: Path to input image
        num_colors: Number of color clusters
        brightness_threshold: Minimum brightness value (0-255)
        saturation_threshold: Minimum saturation value (0-255)
    
    Returns:
        List of RGB color values
    """
    try:
        # Suppress the physical cores warning
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'

        # Read the image
        image = cv2.imread(image_path)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape the image to a 2D array of pixels
        pixels = image_rgb.reshape((-1, 3))

        # Perform k-means clustering for color quantization
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)

        # Get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Filter clusters based on brightness and saturation
        filtered_clusters = []
        for center in cluster_centers:
            # Reshape center to a 1x1 array
            center_reshaped = center.reshape((1, 1, 3))

            # Convert cluster center to HSV
            center_hsv = cv2.cvtColor(center_reshaped.astype(np.uint8), cv2.COLOR_RGB2HSV)[0][0]

            # Check brightness and saturation thresholds
            if center_hsv[2] > brightness_threshold and center_hsv[1] > saturation_threshold:
                filtered_clusters.append(center)

        # Convert the filtered clusters back to uint8
        filtered_clusters = np.array(filtered_clusters, dtype=np.uint8)

        # Convert the colors to BGR for display
        filtered_colors_bgr = cv2.cvtColor(filtered_clusters.reshape((1, -1, 3)), cv2.COLOR_RGB2BGR)

        # Display the filtered colors
        cv2.imshow('Filtered Colors', filtered_colors_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert the filtered colors to a list of RGB values
        filtered_colors_rgb = filtered_clusters.tolist()

        return filtered_colors_rgb

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")

if __name__ == "__main__":
    result = extract_bright_colorful_colors('samples/sample_image.jpg', num_colors=8, 
                                           brightness_threshold=150, saturation_threshold=50)
    print(result)
