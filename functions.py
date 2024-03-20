import cv2
import numpy as np

sobel_v = np.float32([
    [-1, -2, -1],
    [0, 0, 0],
    [+1, +2, +1]
])

sobel_h = np.transpose(sobel_v)

lower_bound = 0
upper_bound = 10 ** 8


def apply_sobel_v(frame):
    frame = np.float32(frame)
    return cv2.filter2D(frame, -1, sobel_v)


def apply_sobel_h(frame):
    frame = np.float32(frame)
    return cv2.filter2D(frame, -1, sobel_h)


def rgb2gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def getBirdsEyeView(frame):
    h, w = frame.shape
    trap_frame = np.zeros((h, w), dtype=np.uint8)

    top_left = (w * 0.41, h * 0.78)
    top_right = (w * 0.57, h * 0.78)
    bot_left = (w * 0.10, h)
    bot_right = (w * 0.80, h)

    # Trapezoid Steps:
    points = np.array([top_right, top_left, bot_left, bot_right], dtype=np.int32)
    cv2.fillConvexPoly(trap_frame, points, 1)
    trap_frame = trap_frame * frame
    cv2.imshow("Trapez: ", trap_frame)
    ##############################

    # Stretched steps:
    screen_points = np.array([(w, 0), (0, 0), (0, h), (w, h)], dtype=np.float32)
    points = np.float32(points)

    magic_matrix = cv2.getPerspectiveTransform(points, screen_points)
    warped = cv2.warpPerspective(trap_frame, magic_matrix, (w, h))
    return warped


def getSobel(frame):
    frame1 = apply_sobel_h(frame)
    frame2 = apply_sobel_v(frame)
    sobel_frame = np.sqrt(frame1 ** 2 + frame2 ** 2)
    return cv2.convertScaleAbs(sobel_frame)


def binarize(frame):
    ret, bin_frame = cv2.threshold(frame, 77, 255, cv2.THRESH_BINARY)
    return bin_frame


def drawLines(frame):
    global left_top, left_bot, right_top, right_bot, right_bot_x, right_top_x, right_top_y, right_bot_y
    h, w = frame.shape
    frame_small = frame.copy()
    frame_small[0: h, 0: round(w * 0.03)] = 0
    frame_small[0: h, round(w * 0.92): w] = 0
    frame_small[round(h * 0.95): h, 0: w] = 0

    # Get coordinates of all white points:
    half_point = w // 2

    leftHalf = frame_small[:, :half_point]
    rightHalf = frame_small[:, half_point:]

    whitesLeft = np.argwhere(leftHalf > 1)
    whitesRight = np.argwhere(rightHalf > 1)

    left_x = whitesLeft[:, 1]
    left_y = whitesLeft[:, 0]
    right_x = whitesRight[:, 1] + half_point
    right_y = whitesRight[:, 0]

    # Filtering some points for better detection:
    threshold = 50
    valid_indices_r = np.abs(right_x - np.mean(right_x)) < threshold
    right_x_filtered = right_x[valid_indices_r]
    right_y_filtered = right_y[valid_indices_r]

    valid_indices_left = np.abs(left_x - np.mean(left_x)) < threshold  # Adjust threshold_left
    left_x_filtered = left_x[valid_indices_left]
    left_y_filtered = left_y[valid_indices_left]

    bl, al = np.polynomial.polynomial.polyfit(left_x_filtered, left_y_filtered, deg=1)

    br, ar = np.polynomial.polynomial.polyfit(right_x_filtered, right_y_filtered, deg=1) if len(right_x) > 0 else (0, 1)
    right_top_y = 0
    right_top_x = (right_top_y - br) / ar if len(right_x) > 0 else half_point

    right_bot_y = h
    right_bot_x = (right_bot_y - br) / ar if len(right_x) > 0 else half_point

    left_top_y = 0
    left_top_x = (left_top_y - bl) / al

    left_bot_y = h
    left_bot_x = (left_bot_y - bl) / al

    if 0 < left_top_x < half_point:
        left_top = int(left_top_x), int(left_top_y)
    if 0 < left_bot_x < half_point:
        left_bot = int(left_bot_x), int(left_bot_y)
    if half_point < right_top_x < w:
        right_top = int(right_top_x), int(right_top_y)
    if half_point < right_bot_x < w:
        right_bot = int(right_bot_x), int(right_bot_y)

    cv2.line(frame_small, left_top, left_bot, (200, 0, 0), 5)
    if len(right_x) != 0:
        cv2.line(frame_small, right_top, right_bot, (100, 0, 0), 5)
    cv2.line(frame_small, (half_point, 0), (half_point, h), (255, 0, 0), 1)

    # text = "Botton Right X: " + str(right_bot_x)
    # cv2.putText(frame_small, text, (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("test", frame_small)

    return


def main_func(frame):
    h, w = frame.shape
    global left_top, left_bot, right_top, right_bot, right_bot_x, right_top_x, right_top_y, right_bot_y
    trap_frame = np.zeros((h, w), dtype=np.uint8)

    top_left = (w * 0.41, h * 0.77)
    top_right = (w * 0.57, h * 0.77)
    bot_left = (w * 0.10, h)
    bot_right = (w, h)

    # Trapezoid Steps:
    points = np.array([top_right, top_left, bot_left, bot_right], dtype=np.int32)
    cv2.fillConvexPoly(trap_frame, points, 1)
    trap_frame_copy = trap_frame.copy()
    cv2.imshow("Trapez alb", trap_frame_copy * 255)
    trap_frame = trap_frame * frame
    cv2.imshow("Trapez: ", trap_frame)
    ##############################

    # Stretched steps:
    screen_points = np.array([(w, 0), (0, 0), (0, h), (w, h)], dtype=np.float32)
    points = np.float32(points)

    magic_matrix = cv2.getPerspectiveTransform(points, screen_points)
    warped = cv2.warpPerspective(trap_frame, magic_matrix, (w, h))
    cv2.imshow("Bird's Eye view", warped)
    ###############################

    # Blurring :
    blured = cv2.blur(warped, ksize=(5, 5))
    cv2.imshow("Blurred", blured)
    # Sobel :
    sobel = getSobel(blured)
    cv2.imshow("Edge detection" , sobel)
    # Binarize :
    binarized = binarize(sobel)
    cv2.imshow("Binarized", binarized)
    frame_small = binarized.copy()
    frame_small[0: h, 0: round(w * 0.03)] = 0
    frame_small[0: h, round(w * 0.92): w] = 0
    frame_small[round(h * 0.95): h, 0: w] = 0

    # Get coordinates of all white points:
    half_point = w // 2

    leftHalf = frame_small[:, :half_point]
    rightHalf = frame_small[:, half_point:]

    whitesLeft = np.argwhere(leftHalf > 1)
    whitesRight = np.argwhere(rightHalf > 1)

    left_x = whitesLeft[:, 1]
    left_y = whitesLeft[:, 0]
    right_x = whitesRight[:, 1] + half_point
    right_y = whitesRight[:, 0]

    # Filtering some points for better detection:
    threshold = 30
    valid_indices_r = np.abs(right_x - np.mean(right_x)) < threshold
    right_x_filtered = right_x[valid_indices_r]
    right_y_filtered = right_y[valid_indices_r]

    valid_indices_left = np.abs(left_x - np.mean(left_x)) < threshold  # Adjust threshold_left
    left_x_filtered = left_x[valid_indices_left]
    left_y_filtered = left_y[valid_indices_left]

    (bl, al) = np.polynomial.polynomial.polyfit(left_x_filtered, left_y_filtered, deg=1) if len(left_x_filtered) != 0 else (0, 1)

    (br, ar) = np.polynomial.polynomial.polyfit(right_x_filtered, right_y_filtered, deg=1) if len(right_x_filtered) != 0 else (0, 1)

    right_top_y = 0
    right_top_x = (right_top_y - br) / ar

    right_bot_y = h
    right_bot_x = (right_bot_y - br) / ar

    left_top_y = 0
    left_top_x = (left_top_y - bl) / al

    left_bot_y = h
    left_bot_x = (left_bot_y - bl) / al


    left_top = int(left_top_x), int(left_top_y)
    left_bot = int(left_bot_x), int(left_bot_y)
    right_top = int(right_top_x), int(right_top_y)
    right_bot = int(right_bot_x), int(right_bot_y)

    cv2.line(frame_small, left_top, left_bot, (200, 0, 0), 5)
    if len(right_x) != 0:
        cv2.line(frame_small, right_top, right_bot, (100, 0, 0), 5)
    cv2.line(frame_small, (half_point, 0), (half_point, h), (255, 0, 0), 1)

    # text = "Botton Right X: " + str(right_bot_x)
    # cv2.putText(frame_small, text, (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Lines", frame_small)

    left_line_frame = np.zeros((h, w), dtype=np.uint8)
    right_line_frame = np.zeros((h, w), dtype=np.uint8)

    cv2.line(left_line_frame, left_top, left_bot, (255, 0, 0), 15)
    cv2.line(right_line_frame, right_top, right_bot, (255, 0, 0), 3)

    magic_matrix_rev = cv2.getPerspectiveTransform(screen_points, points)

    left_line_frame = cv2.warpPerspective(left_line_frame, magic_matrix_rev, (w, h))
    right_line_frame = cv2.warpPerspective(right_line_frame, magic_matrix_rev, (w, h))

    leftCoords = np.argwhere(left_line_frame > 1)
    rightCoords = np.argwhere(right_line_frame > 1)

    return leftCoords, rightCoords
