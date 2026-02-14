import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def extract_bright_colorful_colors(image_path, num_colors=8, brightness_threshold=100, saturation_threshold=30):
    """
    Extract bright and colorful colors from an image using K-means clustering.
    
    Args:
        image_path: Path to input image
        num_colors: Number of color clusters to extract
        brightness_threshold: Minimum brightness value (0-255) for filtering
        saturation_threshold: Minimum saturation value (0-255) for filtering
    
    Returns:
        List of RGB color values sorted by brightness
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
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)

        # Get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Score clusters by brightness and saturation
        scored_clusters = []
        for center in cluster_centers:
            # Reshape center to a 1x1 array
            center_reshaped = center.reshape((1, 1, 3))

            # Convert cluster center to HSV for brightness and saturation analysis
            center_hsv = cv2.cvtColor(center_reshaped.astype(np.uint8), cv2.COLOR_RGB2HSV)[0][0]
            
            brightness = center_hsv[2]
            saturation = center_hsv[1]
            
            # Score based on combination of brightness and saturation
            # Prefer bright and saturated colors, but don't strictly filter
            score = brightness * 0.6 + saturation * 0.4
            
            scored_clusters.append((score, center))

        # Sort by score (brightness + saturation) and take top num_colors
        scored_clusters.sort(reverse=True, key=lambda x: x[0])
        top_clusters = [center for score, center in scored_clusters[:num_colors]]

        # Convert to uint8
        top_clusters = np.array(top_clusters, dtype=np.uint8)

        # Convert the colors to a list of RGB values
        colors_rgb = top_clusters.tolist()

        return colors_rgb

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    result = extract_bright_colorful_colors('samples/sample_image.jpg', num_colors=8, 
                                           brightness_threshold=150, saturation_threshold=50)
    print(result)
