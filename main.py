import cv2

from functions import *

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
new_h = round(900 / 3)
new_w = round(1920 / 4)

while True:
    ret, frame = cam.read()

    if ret is False:
        break

    frame = cv2.resize(frame, (new_w, new_h))
    final = frame.copy()

    gray = rgb2gray(frame)
    cv2.imshow("Grayscale", gray)
    leftCoords, rightCoords = main_func(gray)

    final[leftCoords[:, 0], leftCoords[:, 1]] = (50, 50, 250)
    final[rightCoords[:, 0], rightCoords[:, 1]] = (50, 250, 50)

    # Display images (comment if you dont want to see all of them) :
    # 1: Grayscale Video
    cv2.imshow("Original", frame)

    # 2: Final Image
    cv2.imshow("Final", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
