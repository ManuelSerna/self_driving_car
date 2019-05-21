#*************************************************************************
# OpenCV Python Tutorial - Find Lanes for Self-Driving Cars (Computer Vision Basics Tutorial)
#*************************************************************************
import cv2
import numpy as np

#-------------------------------------------------------------------------
# Make coordinates for getting average slope and intercept
#-------------------------------------------------------------------------
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    return np.array([x1, y1, x2, y2])

#-------------------------------------------------------------------------
# Get average slope-intercept line
#-------------------------------------------------------------------------
def average_slope_intercept(image, lines):
    left_fit = []  # get left lane
    right_fit = []  # get right lane
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        
        # Left: (-) slope
        # Right: (+) slope
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    # Average out all and right left fits. Average slopes and intercepts, so operate vertically (axis = 0).
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])

#-------------------------------------------------------------------------
# Canny algorithm: greyscale, reduce noise, and then canny (i.e. show edges only)
#-------------------------------------------------------------------------
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    return canny_image

#-------------------------------------------------------------------------
# Region of interest: focus on a certain area on an image.
#-------------------------------------------------------------------------
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
     ([(200, height), (1100, height), (550, 250)])
    ])
    
    # Apply triangle shape (defined above) and apply it to a black image with the same dimensions as the original image. So now there is a shape in the black image
    # Note: fillPoly takes in an array of polygons, but here we will just give it one polygon
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

#-------------------------------------------------------------------------
# Display lines
#-------------------------------------------------------------------------
def display_lines(image, lines):
    line_image = np.zeros_like(image)  # same image size as arg image, but black
    
    if lines is not None:
        # draw a line segment connecting two pts. Give image and two points to connect. Give a color (blue). Then give a thickness to the line
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    return line_image

#=========================================================================
# MAIN program to detect lanes in a video
#=========================================================================
cap = cv2.VideoCapture("test2.mp4") # Create a video capture object

while(cap.isOpened()):
    _, frame = cap.read() # get frame capture from video
    
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)    
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
