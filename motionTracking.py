import cv2
import numpy as np

def getNextFrame(vidObj):
    """Takes in the VideoCapture object and reads the next frame, returning one that is half the size
    (Comment out that line if you want fullsize)."""
    ret, frame = vidObj.read()
    print (type(vidObj), type(frame))
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    # ret, gray = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY)
    return frame, gray
# =======
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
# >>>>>>> 4e52f50ae362250c479a0baa939edb82f958850c
#
#     return frame, blur


kernel = np.ones((5,5), np.uint8)
cam = cv2.VideoCapture(0)
cv2.namedWindow('Motion Tracking')
preOrig, prevFrame = getNextFrame(cam)

while True:
    currOrig, currFrame = getNextFrame(cam)
    diff = cv2.absdiff(prevFrame, currFrame)
    ret, gray = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    img_dilation = cv2.dilate(gray, kernel, iterations=1)
    img, contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    for c in contours:
        area = cv2.contourArea(c)

        if(area > 500):
            (x,y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(currOrig, (x,y),(x+w, y+h), (0,255, 0),2)
            cv2.drawContours(currOrig, contours, -1, (0, 255, 0), 3)

    img, contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(diff, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("diff", diff )
    # cv2.imshow("Motion Tracking", cont)
# =======
    cv2.imshow("diff", img_dilation )
    cv2.imshow("Motion Tracking", currOrig)

    x = cv2.waitKey(20)
    c = chr(x & 0xFF)
    if c == "q":
        break
    prevFrame = currFrame

cam.release()
cv2.destroyAllWindows()
