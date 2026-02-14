import numpy as np
import cv2
from matplotlib import pyplot as plt

def detect_edges_canny(input_path, output_path=None, low_threshold=100, high_threshold=200):
    """
    Detect edges using Canny edge detection algorithm.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image (optional)
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
    
    Returns:
        Detected edges as numpy array
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    if output_path:
        cv2.imwrite(output_path, edges)
    
    return edges

def display_edges(input_path, output_path=None):
    """
    Display original image and edge-detected image side by side.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    edges = detect_edges_canny(input_path, output_path)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    display_edges('samples/sample_image.jpg', 'output/canny_edges.png')
