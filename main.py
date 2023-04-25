import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('color_detector_model.h5')

# Define classes
classes = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'violet', 'white', 'yellow']

# Define the range of each color in HSV space
color_ranges = {
    'black': [(0, 0, 0), (180, 255, 30)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'brown': [(0, 50, 20), (20, 255, 200)],
    'green': [(40, 50, 50), (80, 255, 255)],
    'grey': [(0, 0, 50), (180, 50, 255)],
    'orange': [(5, 50, 50), (15, 255, 255)],
    'red': [(-10, 50, 50), (5, 255, 255)],
    'violet': [(130, 50, 50), (160, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'yellow': [(20, 50, 50), (40, 255, 255)]
}

# Define the font for the text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 3

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the center point of the frame
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    # Draw a circle at the center point
    radius = 20
    color = (0, 0, 255)
    thickness = 1
    cv2.circle(frame, (center_x, center_y), radius, color, thickness)

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the color at the center point
    center_color = hsv_frame[center_y, center_x]
    center_color = center_color.reshape(1, 1, 3)

    # Classify the color
    color_name = None
    color_prob = None
    for class_name, (lower, upper) in color_ranges.items():
        if np.all(lower <= center_color) and np.all(center_color <= upper):
            color_name = class_name
            roi = np.zeros((64, 64, 3), dtype=np.uint8)
            roi[:, :, :] = center_color
            roi = cv2.cvtColor(roi, cv2.COLOR_HSV2RGB)
            roi = roi.astype('float32') / 255.0
            pred = model.predict(np.expand_dims(roi, axis=0))
            color_prob = pred[0][classes.index(class_name)]
            break

    # Draw the text overlay
    if color_name is not None:
        text = f'{color_name} ({color_prob:.2f})'
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = center_x - text_size[0] // 2
        text_y = center_y - radius - 100 - text_size[1]
        text_bottom_left = (text_x, text_y)
        color = tuple(int(c) for c in color_ranges[color_name][0])
        cv2.putText(frame, text, text_bottom_left, font, font_scale, color, thickness+2)
        cv2.putText(frame, text, text_bottom_left, font, font_scale, (255, 255, 255), thickness)

        # Set the text color and border color
        text_color = (255, 255, 255)
        border_color = tuple(map(int, color_ranges[color_name][0]))

        # Draw the text with border
        border_thickness = 3
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, border_color, border_thickness+2, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, border_thickness, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Color Detector', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()