import cv2
import numpy as np


def image_stitching(images):
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        print("Image stitching failed!")
        return None


I0='C:/Users/admin/Python Projects/ImageStiching/Images/set1/img1.jpg'
I1='C:/Users/admin/Python Projects/ImageStiching/Images/set1/img2.jpg'
I2='C:/Users/admin/Python Projects/ImageStiching/Images/set1/img3.jpg'

input_img= [I0,I1,I2]

images = [cv2.imread(image) for image in input_img]

stiched_img = image_stitching(images)

if stiched_img is not None:
    cv2.imshow("Stitched Image", stiched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()