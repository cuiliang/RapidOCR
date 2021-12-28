import cv2
import numpy as np
import time

def detect_text(image, show_result=False):

    st = time.time()

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)

    #binary # not used
    # ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    edges = cv2.Canny(gray, 100, 200)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)
    
    
    kernel = np.ones((5,3),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closing', closing)
    cv2.waitKey(0)


    kernel = np.ones((4,4),np.uint8)
    erosion = cv2.erode(closing, kernel, iterations = 1)
    # cv2.imshow('erosion', erosion)
    cv2.waitKey(0)

    #dilation
    kernel = np.ones((5,20), np.uint8)
    img_dilation = cv2.dilate(erosion, kernel, iterations=2)
    # cv2.imshow('dilated',img_dilation)
    cv2.waitKey(0)

    #find contours

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    def func(ctr):
        rect = cv2.boundingRect(ctr)
        n = rect[1]
        n = n - n % 10
        return n, rect[0]
        
    sorted_ctrs = sorted(ctrs, key=func)
    
    sorted_ctrs = [
        cv2.boundingRect(ctr)
        for ctr in sorted_ctrs
    ]
    
    real_sorted_ctrs = []
    while len(sorted_ctrs) != 0:
        x, y, w, h = sorted_ctrs[0]
        top = h / 2 + y
        
        line = []
        for i in range(len(sorted_ctrs)):
            this_x, this_y, this_w, this_h = sorted_ctrs[i]
            this_top = this_h / 2 + this_y
            if abs(top - this_top) < 11:
                line.append(
                    i
                )
        for i in line:
            real_sorted_ctrs.append(
                sorted_ctrs[i]
            )
            sorted_ctrs[i] = None

            
        sorted_ctrs = [
            i
            for i in sorted_ctrs
            if i is not None
        ]
                
                
    contours_list = []
    for i, ctr in enumerate(real_sorted_ctrs):
        # Get bounding box
        x, y, w, h = ctr
        
        limit = 15
        if w < limit or h < limit:
            continue    
        
        contours_list.append(
            [
                x, y, w, h
            ]
        )



    # different form. not used.
    # _contours_list = []
    # for contour in contours_list:
    #     x, y, w, h = contour
    #     _contours_list.append([
    #         x,
    #         y,
    #         x+w,
    #         y+h,
    #     ])


    dt = time.time() - st
    # print("[my_detect_text] time consuming:", dt)

    if show_result:
        for contour in contours_list:
            x, y, w, h = contour
            cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)

        # cv2.imwrite('final_bounded_box_image.png', image)
        cv2.imshow('marked areas',image)
        cv2.waitKey(0)


    cropped_images = []
    for contour in contours_list:
        x, y, w, h = contour
        img = image[y:y+h, x:x+w]
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        cropped_images.append(img)

    return contours_list, cropped_images


if __name__ == '__main__':
    #import image
    image = cv2.imread('in.png')
    contours_list, cropped_images = detect_text(
        image, 
        show_result=True
    )
    print("len(contours_list):", len(contours_list))

