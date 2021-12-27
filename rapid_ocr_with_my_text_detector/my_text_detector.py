import cv2
import numpy as np
import time


def do_something(
    my_image, 
    dilation_kernel=(5, 5),
    output_image=True,
):
    """
    do dilation or find contour
    """
    image_h, image_w = my_image.shape

    #dilation
    kernel = np.ones(dilation_kernel, np.uint8)
    img_dilation = cv2.dilate(my_image, kernel, iterations=1)
    # cv2.imshow('dilated',img_dilation)
    # cv2.waitKey(0)
    
    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    def func(ctr):
        rect = cv2.boundingRect(ctr)
        n = rect[1]
        n = n + (5 - n % 10)
        return n, rect[0]
        
    sorted_ctrs = sorted(ctrs, key=func)

    contours_list = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        # roi = my_image[y:y+h, x:x+w]

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        # cv2.imwrite("segment_no_"+str(i)+".png",roi)
            
        contours_list.append(
            [
                x, y, w, h
            ]
        )
    
    if output_image is True:
        return img_dilation
    else:
        return contours_list


def detect_text(image, show_result=False):

    st = time.time()

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)

    #binary
    ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    edges = cv2.Canny(gray, 100, 200)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)


    img_dilation = do_something(edges, (3, 5), True)
    contours_list = do_something(img_dilation, (1, 5), False)

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

        # cv2.imwrite('final_bounded_box_image.png',image)
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

