import numpy as np
import cv2 as cv

def click_colour(event, x, y, flags, param):
    """
    Registers mouse clicks on an image

    Writes the colour of the clicked pixel to a text file

    Not used in final project
    """
    if event == cv.EVENT_LBUTTONDOWN:
        colors0 = image[y,x,0]
        colors1 = image[y,x,1]
        colors2 = image[y,x,2]
        colors = image[y,x]

        # write colours to a text file 
        text_file = open("colour.txt", "w")

        # text_file.write("%s" % colors)
        text_file.write("%s %s %s" % (colors0, colors1, colors2))
        text_file.close()

def k_means(img, colour_space):
    """
    Performs k-means clustering on an image for colour quantization

    RETURNS: an image after colour quantization; colour values in the corresponding colour space
    """
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters (K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    print("The colours in " + colour_space + " are:\n" + np.array_str(center))

    return res2, center

if __name__ == "__main__":
    # read an image (default colour space is BGR in OpenCV)
    # bgr = cv.imread('sunset.png')
    # bgr = cv.imread('cheetah')
    bgr = cv.imread('sunrise.jpg')
    # bgr = cv.imread('starry_night.jpg')
    # bgr = cv.imread('scream.jpg')

    # k-means in BGR colour space
    res_bgr, colours_bgr = k_means(bgr, "bgr")
    image = res_bgr    

    # OpenCV HSV: hue range is [0,179], saturation range is [0,255], value range is [0,255]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    res_hsv, colours_hsv = k_means(hsv, "hsv")
    # image = res_hsv

    # clear the text file before adding the image colours
    text_file = open("colour.txt", "w")
    text_file.write("")
    text_file.close()

    # add each image colour in HSV colour space to the text file
    for colour in colours_hsv:
        h = colour[0]
        s = colour[1]
        v = colour[2]

        # write colours to a text file
        text_file = open("colour.txt", "a")
        text_file.write("%s %s %s" % (h, s, v) + "\n")
        text_file.close()

    # Display image in BGR space after segmentation
    cv.imshow('k-means',image)

    # cv.setMouseCallback("k-means", click_colour)
    cv.waitKey(0)
    cv.destroyAllWindows()