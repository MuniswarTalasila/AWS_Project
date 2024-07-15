import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from webcolors import HTML4


def list_images(images, cols = 2, rows = 5, cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: List of np.arrays compatible with plt.imshow.
            cols (Default = 2): Number of columns in the figure.
            rows (Default = 5): Number of rows in the figure.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap = cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

# Reading in the test images
test_images = [plt.imread(img) for img in glob.glob('test_images/*.jpg')]
list_images(test_images)

def RGB_color_selection(image):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    # Yellow color mask
    lower_threshold = np.uint8([175, 175,   0])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image

# list_images(list(map(RGB_color_selection, test_images)))

def HSL_color_selection(image):
    """
    Apply color selection to HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # White color mask
    lower_threshold = np.uint8([  0, 200,   0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    # Yellow color mask
    lower_threshold = np.uint8([ 10,   0, 100])
    upper_threshold = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image

# list_images(list(map(HSL_color_selection, test_images)))

def gray_scale(image):
    """
    Grayscale images to remove color information.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# list_images(list(map(gray_scale, list(map(HSL_color_selection, test_images)))), cols=2, rows=5, cmap='gray')

def gaussian_smoothing(image, kernel_size=13):
    """
    Apply Gaussian smoothing to the input image.
        Parameters:
            image: Single channel image.
            kernel_size (Default = 13): Size of the Gaussian kernel.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# list_images(list(map(gaussian_smoothing, list(map(gray_scale, list(map(HSL_color_selection, test_images)))))), cols=2, rows=5, cmap='gray')

def canny_detector(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny Edge Detection algorithm to detect edges in the image.
        Parameters:
            image: Single channel image.
            low_threshold (Default = 50): Low threshold for hysteresis procedure.
            high_threshold (Default = 150): High threshold for hysteresis procedure.
    """
    return cv2.Canny(image, low_threshold, high_threshold)

# list_images(list(map(canny_detector, list(map(gaussian_smoothing, list(map(gray_scale, list(map(HSL_color_selection, test_images)))))))), cols=2, rows=5, cmap='gray')

def region_selection(image):
    """
    Apply region of interest mask to the input image.
        Parameters:
            image: Single channel image.
    """
    mask = np.zeros_like(image)
    match_mask_color = 255
    
    # Defining a four sided polygon to mask
    height, width = image.shape
    region = np.array([[
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]], np.int32)
    
    cv2.fillPoly(mask, region, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

# list_images(list(map(region_selection, list(map(canny_detector, list(map(gaussian_smoothing, list(map(gray_scale, list(map(HSL_color_selection, test_images)))))))))), cols=2, rows=5, cmap='gray')

def hough_transform(image):
    """
    Apply Hough Transform to the input image to detect lane lines.
        Parameters:
            image: Single channel image.
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

hough_lines = [hough_transform(region_selection(canny_detector(gaussian_smoothing(gray_scale(HSL_color_selection(image)))))) for image in test_images]


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    """
    Draw lines on the image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            lines: Lines to be drawn.
            color (Default = [255, 0, 0]): Color of the lines.
            thickness (Default = 2): Thickness of the lines.
    """
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=5):
    """
    Draw lane lines on the image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            lines: Lines to be drawn.
            color (Default = [255, 0, 0]): Color of the lines.
            thickness (Default = 5): Thickness of the lines.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

lane_images = []
for image, lines in zip(test_images, hough_lines):
    lane_images.append(draw_lane_lines(image, lines))

list_images(lane_images)
'''
def frame_processor(image):
    """
    Process the input frame to detect lane lines.
        Parameters:
            image: Single video frame.
    """
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, hough)
    return result

def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
        Parameters:
            test_video: Input video.
            output_video: A video file with detected lane lines.
    """
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False) # type: ignore
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(os.path.join('output_videos', output_video), audio=False, fps=input_video.fps)

HTML("""<video width="960" height="540" controls>  <source src="{0}"></video>""".format("output_videos/solidWhiteRight_output.mp4")) # type: ignore
HTML(""" <video width="960" height="540" controls>  <source src="{0}"></video>""".format("output_videos/solidYellowLeft_output.mp4")) # type: ignore

'''