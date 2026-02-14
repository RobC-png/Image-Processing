# Edge Detector

A comprehensive collection of image processing tools for edge detection, color extraction, and document scanning.

## Features

- **Canny Edge Detection** - Advanced edge detection using OpenCV's Canny algorithm
- **Sobel Edge Detection** - Custom implementation of Sobel operator for edge detection
- **Document Scanner** - Interactive four-point perspective transformation for document scanning
- **Color Extraction** - Extract bright and colorful colors from images using K-means clustering
- **Shading Removal** - Remove shadows and shading while preserving colors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RobC-png/Edge-Detector.git
cd Edge-Detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Canny Edge Detection
```python
from src.canny_edge_detector import detect_edges_canny

edges = detect_edges_canny('path/to/image.jpg', 'output/edges.png')
```

### Sobel Edge Detection
```python
from src.sobel_edge_detector import detect_edges_sobel

detect_edges_sobel('path/to/image.jpg', 'output/sobel.png')
```

### Document Scanner
```python
from src.document_scanner import detect_document

# Click four corners of the document in order, it will be transformed
detect_document('path/to/document.jpg')
```

### Color Extraction
```python
from src.color_extractor import extract_bright_colorful_colors

colors = extract_bright_colorful_colors('path/to/image.jpg', num_colors=8)
print(colors)
```

### Shading Removal
```python
from src.remove_shading import remove_shading_and_keep_colors

remove_shading_and_keep_colors('path/to/image.jpg', 'output/no_shading.jpg')
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pillow
- scikit-learn
- Matplotlib

See `requirements.txt` for full list.

## Project Structure

```
Edge-Detector/
├── src/
│   ├── canny_edge_detector.py      # Canny edge detection
│   ├── sobel_edge_detector.py      # Sobel edge detection
│   ├── document_scanner.py         # Document perspective transformation
│   ├── color_extractor.py          # Color extraction with clustering
│   └── remove_shading.py           # Shading removal
├── samples/                        # Example images
├── output/                         # Output directory for results
├── README.md
└── requirements.txt
```

## License

MIT License

## Author

RobC-png

## Contributing

Feel free to fork this project and submit pull requests for any improvements!
