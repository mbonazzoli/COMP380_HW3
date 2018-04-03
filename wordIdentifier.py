import cv2
import numpy as np

# Locate sign and warp
#   Using corner detection
# Find color
# Feature match using ORB

def locateSign(img):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Computing Harris
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # isolate Harris corner from edge
    dilDist = cv2.dilate(dst, None)
    thresh = 0.01 * dst.max()
    ret, threshDst = cv2.threshold(dilDist, thresh, 255, cv2.THRESH_BINARY)


    # display corners points, test
    disp = np.uint8(threshDst)
    cv2.imshow("Harris", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

