import cv2
import numpy as np

# Function to perform road edge detection and highlight the road edges


def detect_road_edges(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(blurred, 75, 150)

    # Find contours in the edges image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the road edges
    mask = np.zeros_like(frame)

    # Draw the road edges on the mask with a green color
    cv2.drawContours(mask, contours, -1, (0, 255, 0), thickness=2)

    # Combine the original frame with the mask to highlight the road edges
    highlighted_frame = cv2.addWeighted(frame, 1, mask, 0.7, 0)

    return highlighted_frame


# Create a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Perform road edge detection and highlighting on the frame
    highlighted_frame = detect_road_edges(frame)

    # Display the resulting frame with highlighted road edges
    cv2.imshow('Road Edge Detection', highlighted_frame)

    # Check if the user pressed 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
