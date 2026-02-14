#!/usr/bin/env python3
"""
Edge Detector - Main Program
A simple program to use various image processing functions.
"""

import os
import sys
import cv2
from src.canny_edge_detector import detect_edges_canny, display_edges
from src.sobel_edge_detector import detect_edges_sobel
from src.color_extractor import extract_bright_colorful_colors
from src.remove_shading import remove_shading_and_keep_colors
from src.document_scanner import detect_document


def validate_image(image_path):
    """Validate that an image can be read by cv2."""
    img = cv2.imread(image_path)
    if img is None:
        return False, None
    return True, img


def print_menu():
    """Display the main menu."""
    print("\n" + "="*50)
    print("        EDGE DETECTOR - IMAGE PROCESSOR")
    print("="*50)
    print("1. Canny Edge Detection")
    print("2. Sobel Edge Detection")
    print("3. Extract Bright & Colorful Colors")
    print("4. Remove Shading from Image")
    print("5. Document Scanner (Select Points)")
    print("6. Exit")
    print("="*50)


def get_image_path():
    """Get image path from user with validation."""
    while True:
        path = input("Enter image path: ").strip()
        if not os.path.exists(path):
            print(f"✗ File not found: '{path}'")
            continue
        
        # Try to load the image with cv2 to verify it's a valid image
        is_valid, img = validate_image(path)
        if not is_valid:
            print(f"✗ Cannot read image: '{path}'")
            print("  - File may be corrupted")
            print("  - Format may not be supported (try .jpg, .png, .bmp, .tiff)")
            print("  - Check file permissions")
            continue
        
        print(f"✓ Image loaded successfully ({img.shape[1]}x{img.shape[0]} pixels)")
        return path


def get_output_path(default_name):
    """Get output path from user."""
    path = input(f"Enter output path (default: output/{default_name}): ").strip()
    if not path:
        path = f"output/{default_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else "output", exist_ok=True)
    return path


def option_canny():
    """Handle Canny edge detection."""
    print("\n--- Canny Edge Detection ---")
    input_path = get_image_path()
    output_path = get_output_path("canny_edges.png")
    low_threshold = input("Enter low threshold (default 100): ").strip()
    high_threshold = input("Enter high threshold (default 200): ").strip()
    
    low_threshold = int(low_threshold) if low_threshold else 100
    high_threshold = int(high_threshold) if high_threshold else 200
    
    try:
        # Verify image can be read before processing
        is_valid, img = validate_image(input_path)
        if not is_valid:
            print(f"✗ Cannot process image: {input_path}")
            return
        
        detect_edges_canny(input_path, output_path, low_threshold, high_threshold)
        print(f"✓ Edge detection complete! Saved to: {output_path}")
        
        show_display = input("Display result? (y/n): ").strip().lower()
        if show_display == 'y':
            display_edges(input_path, output_path, show_plot=True)
    except AssertionError as e:
        print(f"✗ Error: {e}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def option_sobel():
    """Handle Sobel edge detection."""
    print("\n--- Sobel Edge Detection ---")
    input_path = get_image_path()
    output_path = get_output_path("sobel_edges.png")
    
    try:
        # Verify image can be read before processing
        is_valid, img = validate_image(input_path)
        if not is_valid:
            print(f"✗ Cannot process image: {input_path}")
            return
        
        detect_edges_sobel(input_path, output_path)
        print(f"✓ Sobel edge detection complete!")
        print("  Output files saved in output/ directory")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def option_color_extract():
    """Handle color extraction."""
    print("\n--- Extract Bright & Colorful Colors ---")
    input_path = get_image_path()
    num_colors = input("Enter number of colors to extract (default 8): ").strip()
    num_colors = int(num_colors) if num_colors else 8
    
    try:
        # Verify image can be read before processing
        is_valid, img = validate_image(input_path)
        if not is_valid:
            print(f"✗ Cannot process image: {input_path}")
            return
        
        colors = extract_bright_colorful_colors(input_path, num_colors=num_colors)
        print(f"✓ Extracted {len(colors)} colors!")
        print(f"Colors (RGB): {colors}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def option_remove_shading():
    """Handle shading removal."""
    print("\n--- Remove Shading from Image ---")
    input_path = get_image_path()
    output_path = get_output_path("no_shading.jpg")
    num_colors = input("Enter number of colors (default 8): ").strip()
    brightness = input("Enter brightness threshold (default 150): ").strip()
    
    num_colors = int(num_colors) if num_colors else 8
    brightness = int(brightness) if brightness else 150
    
    try:
        # Verify image can be read before processing
        is_valid, img = validate_image(input_path)
        if not is_valid:
            print(f"✗ Cannot process image: {input_path}")
            return
        
        remove_shading_and_keep_colors(input_path, output_path, num_colors, brightness)
        print(f"✓ Shading removed! Saved to: {output_path}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def option_document_scan():
    """Handle document scanning."""
    print("\n--- Document Scanner ---")
    input_path = get_image_path()
    
    try:
        # Verify image can be read before processing
        is_valid, img = validate_image(input_path)
        if not is_valid:
            print(f"✗ Cannot process image: {input_path}")
            return
        
        print("Instructions: Click on 4 corners of the document to scan it.")
        print("Points should be selected in order: top-left, top-right, bottom-right, bottom-left")
        detect_document(input_path)
        print("✓ Document scanning complete! Saved to: output/transformed_image.png")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def main():
    """Main program loop."""
    os.makedirs("output", exist_ok=True)
    
    while True:
        print_menu()
        choice = input("Select an option (1-6): ").strip()
        
        if choice == '1':
            option_canny()
        elif choice == '2':
            option_sobel()
        elif choice == '3':
            option_color_extract()
        elif choice == '4':
            option_remove_shading()
        elif choice == '5':
            option_document_scan()
        elif choice == '6':
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("✗ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
