import cv2
import numpy as np

# Locate sign and warp
#   Using corner detection
# Find color
# Feature match using ORB

hatch = cv2.imread("Images/hatchTest.JPG")
door = cv2.imread("Images/doorTest.JPG")
exit = cv2.imread("Images/exitTest.JPG")

reference_images = {'door': door, 'hatch': hatch, 'exit': exit}

sign_names = ['door', 'hatch', 'exit']

# vert_stack = np.vstack((hatch, door, exit))
# vert_concat = np.concatenate((hatch, door, exit), axis=1)
#
# cv2.imshow("concat", vert_concat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# locates the edges within a scene
def locateSign(img):
    """Locates the corners within the image"""
    # TODO:  find the outer corners of the sign and affine warp to isolate
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Computing Harris
    dst = cv2.cornerHarris(gray, 10, 3, 0.04)

    # isolate Harris corner from edge
    dilDist = cv2.dilate(dst, None)
    thresh = 0.01 * dst.max()
    ret, threshDst = cv2.threshold(dilDist, thresh, 255, cv2.THRESH_BINARY)

    # display corners points, test
    disp = np.uint8(threshDst)
    cv2.imshow("Harris", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compareImages(ref, img):
    """Compares two images and returns the number of matches between the two images """
    threshold = 50.0

    orb = cv2.ORB_create()
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Find key features
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    # Match key features
    matches = bfMatcher.match(des1, des2)
    matches.sort(key=lambda x: x.distance)  # sort by distance

    # Find matches with distance less than threshold
    for i in range(len(matches)):
        if matches[i].distance > threshold:
            break
    # return matches with distance less than threshold
    return len(matches[:i])


def findMatch():
    vidCap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        x = cv2.waitKey(10)
        char = chr(x &0xFF)

        if char == 'q':
            cv2.destroyAllWindows()
            vidCap.release()
        ret, vid = vidCap.read()
        cv2.imshow("Sign Match", vid)

        maxMatches = 0
        signMatch = ''

        for name in sign_names:
            numMatches = compareImages(reference_images[name], vid)
            print(name + " " + str(numMatches))
            if numMatches > maxMatches:
                maxMatches = numMatches
                signMatch = name

        print("Final Match " + signMatch + " " + str(maxMatches))

# compareImages(img1, img2)

# locateSign(img1)

findMatch()