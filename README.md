# Edge Detector

A collection of image processing tools I built while messing around with python.

## What's in here

- **Canny Edge Detection** - OpenCV's Canny algorithm wrapped in a simple CLI
- **Sobel Edge Detection** - Custom edge detection via gradient calculation
- **Document Scanner** - Click 4 corners of a document, get a perspective-corrected crop
- **Color Extraction** - K-means clustering to pull out the most vibrant colors
- **Shading Removal** - Remove dark shadows while keeping colors intact

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

Python 3.7+, OpenCV, NumPy, Pillow, scikit-learn, Matplotlib
