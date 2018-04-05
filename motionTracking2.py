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
# =======
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
# >>>>>>> 4e52f50ae362250c479a0baa939edb82f958850c
#
#     return frame, blur
vid1 = cv2.VideoCapture(0)
kernel = np.ones((5,5), np.uint8)
cv2.namedWindow('Motion Tracking')
ret, frame = vid1.read()
twoImage = [frame, frame]

while True:
    ret2, image = vid1.read()
    twoImage.append(image)
    twoImage.pop(0)

    pic1 = twoImage[0]
    pic2 = twoImage[1]

    overGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overGray = cv2.GaussianBlur(overGray, (25, 25), 0)
    gray1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (25, 25), 0)
    gray2 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (25, 25), 0)

    diffMotion = cv2.absdiff(gray1, gray2)
    mothresh1 = cv2.threshold(diffMotion, 80, 255, cv2.THRESH_BINARY)[1]
    mothresh1 = cv2.dilate(mothresh1, None, iterations = 2)
    mothresh1 = cv2.threshold(diffMotion, 80, 255, cv2.THRESH_BINARY)[1]
    diffMotion2 = cv2.absdiff(gray1, gray2)
    mothresh2 = cv2.threshold(diffMotion2, 80, 255, cv2.THRESH_BINARY)[1]
    mothresh2 = cv2.dilate(mothresh2, None, iterations=2)
    mothresh2 = cv2.threshold(diffMotion2, 80, 255, cv2.THRESH_BINARY)[1]

    (_, mycontours, _) = cv2.findContours(mothresh1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in mycontours:
        if cv2.contourArea(c) > 600:
            continue
        frame = pic2

    (_, mycontours, _) = cv2.findContours(mothresh2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, mycontours, -1, (128, 128, 128), 2)
    for c in mycontours:
        if cv2.contourArea(c) < 700:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # currOrig, currFrame = getNextFrame(cam)
    # diff = cv2.absdiff(prevFrame, currFrame)
    # ret, gray = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # img_dilation = cv2.dilate(gray, kernel, iterations=1)
    # img, contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(currOrig, contours, -1, (0, 255, 0), 3)




    # for c in contours:
    #     area = cv2.contourArea(c)
    #     if(area > 200):
    #         (x,y, w, h) = cv2.boundingRect(c)
    #         cv2.rectangle(currOrig, (x,y),(x+w, y+h), (0,255, 0),2)

# <<<<<<< HEAD
#     img, contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cont = cv2.drawContours(diff, contours, -1, (0, 255, 0), 3)
#     cv2.imshow("diff", diff )
#     cv2.imshow("Motion Tracking", cont)
# =======
    cv2.imshow("diff", img_dilation )
    cv2.imshow("Motion Tracking", currOrig)

    x = cv2.waitKey(20)
    c = chr(x & 0xFF)
    if c == "q":
        break

cam.release()
cv2.destroyAllWindows()
