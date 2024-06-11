import cv2
import numpy as np

I1='C:/Users/admin/Python Projects/ImageStiching/Images/set4/nature.jpg'

img = cv2.imread(I1)

# cv2.imread() -> takes an image as an input
h, w, channels = img.shape

splits=3
half = w // splits

overlap1_1=np.int(np.round(half*0.05))

# this will be the first column
rt_edge=half+overlap1_1

img1 = img[:, :rt_edge]


# [:,:half] means all the rows and
# all the columns upto index half



1
for i in range (splits):
    img1 = img[:, i*half: i*half+rt_edge]
    filename=('C:/Users/admin/Python Projects/ImageStiching/Images/set4/'+'img'+np.str(i+1)+'.jpg')
    cv2.imwrite(filename, img1)

