def Changebackgroundvideo1(img2):
    import cv2
    from cvzone.SelfiSegmentationModule import SelfiSegmentation
    segmentor = SelfiSegmentation()
   # img2=cv2.imread('C:/Users/admin/Python Projects/ImageStiching/Images/Change Background/BackgroundImage.jpg')
    imgBg = cv2.resize(img2, (320, 240))
    # open camera
    cap = cv2.VideoCapture(0)
    capVideo = cv2.VideoCapture("C:/Users/admin/Python Projects/FogRemoval/Images/VID_20211229_073627912.mp4")
    while True:
        # read image
        ret, img = cap.read()
        # read video frame
        ret, videoFrame = capVideo.read()
        if not ret:
            break
        # resize frames to 320 x 240
        img = cv2.resize(img, (320, 240))
        #videoFrame = cv2.resize(videoFrame, (320, 240))
        videoFrame = cv2.resize(imgBg, (320, 240))
        imgBgVideo = segmentor.removeBG(img, videoFrame)
        # show both images
        # cv2.imshow('office',img)
        # cv2.imshow('Black Adam',videoFrame)
        cv2.imshow('Change Background',imgBgVideo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # close camera
    cap.release()
    cv2.destroyAllWindows()


Changebackgroundvideo1()









































