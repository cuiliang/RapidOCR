
import cv2
import numpy as np

#import image
image = cv2.imread('my_input1.png')
#cv2.imshow('my_input image',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('grayscale',gray)
#cv2.waitKey(0)

#binary
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('binary',thresh)
#cv2.waitKey(0)

#dilation
kernel = np.ones((3,20), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#find contours
#ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for i, ctr in enumerate(ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    # cv2.imwrite("segment_no_"+str(i)+".png",roi)
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (90, 0, 255),
        2
    )
    cv2.waitKey(0)

# cv2.imwrite('final_bounded_box_image.png',image)
cv2.imshow('marked areas',image)
cv2.waitKey(0)