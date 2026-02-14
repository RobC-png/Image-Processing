import cv2
import numpy as np

# Global variables to store user-defined points
user_points = []

def mouse_callback(event, x, y, flags, param):
    global user_points, image, original_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Capture the clicked point
        user_points.append((x, y))

        # Draw a circle at the clicked point on the image
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", image)

        # If four points are selected, proceed to perspective transformation
        if len(user_points) == 4:
            rect = order_points(np.array(user_points))
            transformed_image = four_point_transform(original_image, rect)
            cv2.imwrite("output/transformed_image.png", transformed_image)
            cv2.destroyAllWindows()

def detect_document(image_path):
    global user_points, image, original_image

    # Step 1: Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Step 3: Create a window to select user-defined points
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", mouse_callback)

    # Step 4: Display the image and wait for user to click four points
    cv2.imshow("Select Points", image)
    cv2.waitKey(0)

def four_point_transform(image, pts):
    # Get the transformed rectangle's width and height
    widthA = np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2))
    widthB = np.sqrt(((pts[1][0] - pts[0][0]) ** 2) + ((pts[1][1] - pts[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2))
    heightB = np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Calculate the perspective transform matrix and warp the image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def order_points(pts):
    # Sort the points based on their x and y coordinates
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Example usage
if __name__ == "__main__":
    detect_document('samples/sample_image.jpg')  # Update the image path
