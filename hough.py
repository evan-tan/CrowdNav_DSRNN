import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, draw, feature, io, transform, util
from skimage.morphology import skeletonize


def test_skimage(img):
    # img = io.imread("fence.png")
    # print(img.shape)
    # img2 = feature.canny(img2, sigma=1)

    img = util.invert(img)
    img = skeletonize(img)
    # grayscale
    img2 = color.rgb2gray(img)
    io.imsave("hough/skimage-canny.png", util.img_as_uint(img2))
    lines = transform.probabilistic_hough_line(img2, line_length=10, line_gap=3)

    # transform.hough_circle()
    # transform.hough_circle_peaks()

    plt.figure(figsize=(10, 10))
    i = plt.imshow(img2)
    # i.set_cmap("hot")
    plt.axis("off")

    for p0, p1 in lines:
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-")

    plt.savefig("./hough_skimage.png", bbox_inches="tight")


def test_opencv(img):
    # img = cv2.imread("fence.png")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sigma = 0.3
    v = np.median(img2)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    img2 = cv2.GaussianBlur(img2, (7, 7), 3)
    img2 = cv2.Canny(img2, lower, upper)

    lines = cv2.HoughLinesP(img2, 1, np.pi / 180, 10, 3, 2)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite("./hough_opencv.png", img)
