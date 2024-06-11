import cv2
import numpy as np

splt_max=3
I1='C:/Users/admin/Python Projects/ImageStiching/Images/set1/img2.jpg'
I2='C:/Users/admin/Python Projects/ImageStiching/Images/set1/img3.jpg'
I3='C:/Users/admin/Python Projects/ImageStiching/Images/set1/img3.jpg'

# Load the images
img1_original = cv2.imread(I1)

img1 = cv2.imread(I1)
img2 = cv2.imread(I2)
img3 = cv2.imread(I3)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)



# c=800
# r=500
# img1_original=cv2.resize(img1_original, (r,c))
# img1=cv2.resize(img1, (r,c))
# img2=cv2.resize(img2, (r,c))
# img3=cv2.resize(img3, (r,c))


# img1=cv2.resize(image1, (300,397))
# img2=cv2.resize(image2, (300,397))


def detect_and_match_features(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    #keypoints3, descriptors3 = orb.detectAndCompute(img3, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints1, keypoints2, matches

def estimate_homography(keypoints1, keypoints2, matches, threshold=3):
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
    return H, mask

def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2



def blend_images(img1, img2):
    mask = np.where(img1 != 0, 1, 0).astype(np.float32)
    blended_img = img1 * mask + img2 * (1 - mask)
    if mask[1,1,1]==1:
         blended_img = img2 * mask + img1 * (1 - mask)


    return blended_img.astype(np.uint8)
no_images=1
while no_images<splt_max:
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    no_images=no_images+1
    keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)
    print('keypoints:',len(matches))

    H, mask = estimate_homography(keypoints1, keypoints2, matches)
    warped_img = warp_images(img2, img1, H)
    #cv2.imshow('warped_img', warped_img)
    img1=cv2.resize(img1, (warped_img.shape[1],warped_img.shape[0]))
    output_img = blend_images(warped_img,img1)
    #output_img = blend_images(img1,warped_img)
    cv2.imshow('output', output_img)
    cv2.waitKey(0)
    img1=[]
    img1=output_img
    img2=[]
    img2=img3
    img2 = cv2.resize(img2, (output_img.shape[1], output_img.shape[0]))




#img1_original=cv2.resize(img1_original, (img2.shape[1],img2.shape[0]))
#numpy_horizontal = np.hstack((img1_original,img2, output_img))

#cv2.imshow('output', output_img)

#cv2.imshow('Numpy Horizontal', numpy_horizontal)

#cv2.imshow('Stitched Image', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('stitched_image.jpg', output_img)