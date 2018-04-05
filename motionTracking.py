import cv2
import numpy as np


#ORB
# # Some versions of OpenCV need this to fix a bug
# # cv2.ocl.setUseOpenCL(False)
#
# img = cv2.imread("TestImages/Puzzle1.jpg")
#
# # create an ORB object, that can run the ORB algorithm.
# orb = cv2.ORB_create()  # some versions use cv2.ORB() instead
#
# keypts, des = orb.detectAndCompute(img, None)
#
# img2 = cv2.drawKeypoints(img, keypts, None, (255, 0, 0), 4)
# cv2.imshow("Keypoints 1", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#FAST
# import cv2
#
# img = cv2.imread("TestImages/shops.jpg")
# cv2.imshow("Original 1", img)
#
# # create a FAST object, that can run the FAST algorithm.
# fast = cv2.FastFeatureDetector_create()
# # detect features
# keypts = fast.detect(img, None)
# img2 = cv2.drawKeypoints(img, keypts, None, (255, 0, 0), 4)
# cv2.imshow("Keypoints 1", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Matching Features
# # Some versions of OpenCV need this to fix a bug
# cv2.ocl.setUseOpenCL(False)
#
# img1 = cv2.imread("TestImages/Coins1.jpg")
# img2 = cv2.imread("TestImages/DollarCoin.jpg")
#
# orb = cv2.ORB_create()
# bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# matches = bfMatcher.match(des1, des2)
# matches.sort(key = lambda x: x.distance)  # sort by distance
#
#
# # draw matches with distance less than threshold
# for i in range(len(matches)):
#     if matches[i].distance > 50.0:
#         break
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:i], None)
# cv2.imshow("Matches", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#FLANN Matching
# FLANN_INDEX_LSH = 6
# indexParams= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# searchParams = dict(checks=50)
#
# img1 = cv2.imread("TestImages/Coins1.jpg")
# img2 = cv2.imread("TestImages/DollarCoin.jpg")
#
# orb = cv2.ORB_create()
# flanner = cv2.FlannBasedMatcher(indexParams, searchParams)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# matches = flanner.match(des1, des2)
# matches.sort(key = lambda x: x.distance)  # sort by distance
#
#
# # draw matches with distance less than threshold
# for i in range(len(matches)):
#     if matches[i].distance > 50.0:
#         break
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:i], None)
# cv2.imshow("Matches", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def getNextFrame(vidObj):
    """Takes in the VideoCapture object and reads the next frame, returning one that is half the size
    (Comment out that line if you want fullsize)."""
    ret, frame = vidObj.read()
    print (type(vidObj), type(frame))
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    # ret, gray = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY)
    return frame, gray

cam = cv2.VideoCapture(0)
cv2.namedWindow('Motion Tracking')
preOrig, prevFrame = getNextFrame(cam)

while True:
    currOrig, currFrame = getNextFrame(cam)
    diff = cv2.absdiff(prevFrame, currFrame)

    img, contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(diff, contours, -1, (0, 255, 0), 3)
    cv2.imshow("diff", diff )
    cv2.imshow("Motion Tracking", cont)

    x = cv2.waitKey(20)
    c = chr(x & 0xFF)
    if c == "q":
        break
    prevFrame = currFrame

cam.release()
cv2.destroyAllWindows()
