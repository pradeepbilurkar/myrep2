#import PIL
from tkinter import *
import tkinter.font as fnt
#import tkinter as tk
from PIL import Image
from PIL import ImageTk
#from PIL import ImageFilter
from tkinter import filedialog
# import cv2
# from skimage import filters
# import skimage
# import mahotas
# import mahotas.demos
import numpy as np
from Gradio_laneDeparture import lanedeparture
from GradioFace_Identification import detect_face
# from scipy.fftpack import fft2, ifft2
import cv2
# from psf2otf import psf2otf
from Gradio_Rain_Smoothing import L0Smoothing
from Gradio_Object_detection import gradio_object
#from Change_Background_Video import Changebackgroundvideo1

#from t_kinter import video_stream
#from Gradio_Language_Translator import my_translator


global panelA,panelB,panelC,panelD,img1,img2,img3
img1=[]
img2=[]
img3=[]

def uploadImage_backup():
    global panelA, panelB, img2
    import cv2
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image = cv2.resize(image, (250, 350))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    image1 = ImageTk.PhotoImage(image1)
    panelA = Label(image=image1, borderwidth=5, relief="sunken")
    panelA.image = image1
    panelA.grid(row=5, column=2, rowspan=15,  padx=0, pady=0)
    panelB.image = None
    panelB.grid(row=5, column=4, rowspan=15, padx=0, pady=0)
    img2 = image
    return img2

def uploadImage1():
    global panelA, panelB, panelC, panelD, img1
    try:
        panelA.config(image='')
    except:
        1
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image_original = image
    image = cv2.resize(image, (250, 350))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    image1 = ImageTk.PhotoImage(image1)
    panelA = Label(image=image1, borderwidth=1, relief="sunken")
    panelA.image = image1
    panelA.grid(row=5, column=1, rowspan=15)
    img1 = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    return img1


def uploadImage2():
    global panelA, panelB, panelC, panelD,img2
    try :
        panelB.config(image='')
    except:
        1
    f_types = [('Jpg Files', '*.jpg'), ('PNG Fi'
                                        'les', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image_original = image
    image = cv2.resize(image, (250, 350))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    image1 = ImageTk.PhotoImage(image1)
    panelB = Label(image=image1, borderwidth=1, relief="sunken")
    panelB.image = image1
    panelB.grid(row=5, column=2, rowspan=15, padx=0, pady=0)
    img2 = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    return img2


def uploadImage3():
    global panelA, panelB, panelC, panelD,img3
    try:
        panelC.config(image='')
    except:
        1
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image_original = image
    image = cv2.resize(image, (250, 350))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    image1 = ImageTk.PhotoImage(image1)
    panelC= Label(image=image1, borderwidth=1, relief="sunken")
    panelC.image = image1
    panelC.grid(row=5, column=3, rowspan=15, padx=0, pady=0)
    img3 = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    return img3


def display_output(output_image):
    global   panelA, panelB, panelC, panelD,img3
    import cv2
    try:
        panelD.config(image='')
    except:
        1
    output_image = cv2.resize(output_image, (250, 350))
    output_image1 = Image.fromarray(output_image)
    output_image1 = ImageTk.PhotoImage(output_image1)
    panelD = Label(image=output_image1, borderwidth=1, relief="sunken")
    panelD.image = output_image1
    panelD.grid(row=5, column=4, rowspan=15, padx=0, pady=0)
#

def display_Video_output(output_image):
    global   panelA, panelB, panelC, panelD,img3
    import cv2
    output_image = cv2.resize(output_image, (250, 350))
    output_image1 = Image.fromarray(output_image)
    output_image1 = ImageTk.PhotoImage(output_image1)
    panelD = Label(image=output_image1, borderwidth=1, relief="sunken")
    panelD.image = output_image1
    panelD.grid(row=5, column=4, rowspan=15, padx=0, pady=0)
    1
def open_camera():
    # Capture the video frame by frame
    from tkvideo import tkvideo
    global panelA, panelB, panelC, panelD, img3
    vid =cv2.VideoCapture("C:/Users/admin/Python Projects/ImageStiching/output.mp4")
    width, height = (250, 350)
    # Set the width and height
    i=0
    while(True):
        i=i+1
        print (i)
        _, frame = vid.read()
        display_Video_output(frame)


# Create a button to open the camera in GUI app






def remove():
    global panelA, panelB,panelC,panelD
    try:
        panelA.config(image='')
        panelA.image = None
        panelA.destroy()

    except:
        1

    try:
        panelB.config(image='')
        panelB.image = None
        panelB.destroy()
    except:
        1
    try:
        panelC.config(image='')
        panelC.image = None
        panelC.destroy()
    except:
        1

    try:
        panelD.config(image='')
        panelD.image = None
        panelD.destroy()
    except:
        1

def remove_output_Image():
    try:
        panelD.config(image='')
        panelD.image = None
        panelD.destroy()
    except:
        1



# def my_translator():
#     from Gradio_Language_Translator import my_translator
#     my_translator()

def change_background():
    remove_output_Image()
    import cv2
    # import numpy as np
    # from PIL import Image
    # from matplotlib import cm
    from rembg import remove
    try:
        panelD.config(image='')
        panelD.image = None
        panelD.destroy()
    except:
        1
    try:
        panelD.config(image='')
        panelD.image = None
        panelD.destroy()
    except:
        1

    # if img1 is None or img2 is None:
    #     return None, {'error': 'Error loading images. Check file paths.'}

    imgages = [img1, img2]

    imgages[1] = cv2.resize(imgages[1], [imgages[0].shape[1], imgages[0].shape[0]])

    imgages[0] = remove(imgages[0])

    imgages[0] = cv2.cvtColor(imgages[0], cv2.COLOR_BGR2RGB)
    imgages[1] = cv2.cvtColor(imgages[1], cv2.COLOR_BGR2RGB)
    gray_img1 = cv2.cvtColor(imgages[0], cv2.COLOR_RGB2GRAY)
    _, alpha = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY)
    alpha_inv = cv2.bitwise_not(alpha)
    img1_region = cv2.bitwise_and(imgages[0], imgages[0], mask=alpha)
    img2_region = cv2.bitwise_and(imgages[1], imgages[1], mask=alpha_inv)
    blended_image = cv2.addWeighted(img1_region, 1, img2_region, 1, 0)
    result = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
    display_output(result)
    return result


def stitch():
    remove_output_Image()
    import cv2
    stitcher = cv2.Stitcher_create()
    if len(img3) ==0:
       imgages=[img1, img2]
    else:
        imgages = [img1, img2, img3]
    status, stitched_image = stitcher.stitch(imgages)

    if status == cv2.Stitcher_OK:
        stitched_image1 = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('C:/Users/admin/Python Projects/ImageStiching/OutputImages/stitchedImage.jpg', stitched_image1)
        display_output(stitched_image)
        return stitched_image
    else:
        print("Image stitching failed!")
        return None

def Remove_Background():
    remove_output_Image()
    from rembg import remove
    try:
        img_rem = remove(img1)
        display_output(img_rem)
    except:
        1
    try:
        img_rem = remove(img2)
        display_output(img_rem)
    except:
        1
    try:
        img_rem = remove(img3)
        display_output(img_rem)
    except:
        1
    display_output(img_rem)
    return img_rem

def Remove_Fog():
    remove_output_Image()
    import image_dehazer
    HazeCorrectedImg, haze_map = image_dehazer.remove_haze(img1, showHazeTransmissionMap=False)
    #HazeCorrectedImg = L0Smoothing(HazeCorrectedImg)
    display_output(HazeCorrectedImg)
    return HazeCorrectedImg

def Remove_Rain():
    remove_output_Image()
    out_image = L0Smoothing(img1)
    # import image_dehazer
    # HazeCorrectedImg, haze_map = image_dehazer.remove_haze(out_image, showHazeTransmissionMap=False)
    # return HazeCorrectedImg
    display_output(out_image)
    return out_image

def Object_Detection():
    remove_output_Image()
    gradio_object()

def Face_Detection():
    remove_output_Image()
    detect_face()

def DMS_new():
    remove_output_Image()
    1
    from Gradio_DMS_new import*
    facial_processing()

def Changebackgroundvideo():
    #import cv2
    from cvzone.SelfiSegmentationModule import SelfiSegmentation
    #from cvzone.SelfiSegmentationModule import SelfiSegmentation
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
        # videoFrame = cv2.resize(videoFrame, (320, 240))
        #videoFrame = cv2.resize(imgBg, (320, 240))
        #imgBgVideo = segmentor.removeBG(img, videoFrame)
        imgBgVideo = segmentor.removeBG(img, imgBg)
        #cv2image = cv2.cvtColor(imgBgVideo, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(imgBgVideo)
        imgtk = ImageTk.PhotoImage(image=img)
        panelD = Label(image=imgtk, borderwidth=1, relief="sunken")
        panelD.image = imgtk
        panelD.grid(row=5, column=4, rowspan=15, padx=0, pady=0)
        panelD.configure(image=imgtk)
        panelD.after(1, Changebackgroundvideo)


        #imgBgVideo = segmentor.removeBG(img, videoFrame)
        cv2.imshow('Change Background', imgBgVideo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # close camera
    cap.release()
    cv2.destroyAllWindows()


def Changebackgroundvideo_Video():
    # Capture from camera
    cap = cv2.VideoCapture(0)

    # function for video streaming
    def video_stream():
        _, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        panelD = Label(image=imgtk, borderwidth=1, relief="sunken")
        panelD.image = imgtk
        panelD.grid(row=5, column=4, rowspan=15, padx=0, pady=0)
        panelD.imgtk = imgtk
        panelD.configure(image=imgtk)
        panelD.after(10, video_stream)

    video_stream()
    root.mainloop()





def grayscale():
    remove_output_Image()
    grayimg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    display_output(grayimg)
    return grayimg

def negative():
    remove_output_Image()
    neg = 255 - img1
    display_output(neg)
    return neg

def threshold():
    remove_output_Image()
    image = grayscale()
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    display_output(thresh)
    return thresh

def redext():
    remove_output_Image()
    row, col, plane = img1.shape
    red = np.zeros((row, col, plane), np.uint8)
    red[:, :, 0] = img1[:, :, 0]
    display_output(red)
    return re

def greenext():
    remove_output_Image()
    row, col, plane = img1.shape
    green = np.zeros((row, col, plane), np.uint8)
    green[:, :, 1] = img1[:, :, 1]
    display_output(green)
    return green

def blueext():
    remove_output_Image()
    row, col, plane = img1.shape
    blue = np.zeros((row, col, plane), np.uint8)
    blue[:, :, 2] = img1[:, :, 2]
    display_output(blue)
    return blue


def edge():
    remove_output_Image()
    img1 = threshold()
    edged = cv2.Canny(img1, 50, 100)
    display_output(edged)
    return edged


def skeleton():
    remove_output_Image()
    img1 = threshold()
    skel = np.zeros(img1.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open = cv2.morphologyEx(img1, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img1, open)
        eroded = cv2.erode(img1, element)
        skel = cv2.bitwise_or(skel, temp)
        img1 = eroded.copy()
        if cv2.countNonZero(img1) == 0:
            break
    display_output(skel)
    return skel

def denoise():
    remove_output_Image()
    denoise = cv2.fastNlMeansDenoisingColored(img1, None, 5, 5, 7, 21)
    display_output(denoise)
    return denoise

def sharp():
    remove_output_Image()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img1, ddepth=-1, kernel=kernel)
    display_output(sharpened)
    return sharpened

def histo():
    remove_output_Image()
    histogram = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    histogram[:, :, 0] = cv2.equalizeHist(histogram[:, :, 0])
    histogram = cv2.cvtColor(histogram, cv2.COLOR_YUV2BGR)
    display_output(histogram)
    return histogram


def powerlawtrans():
    remove_output_Image()
    gammaplt = np.array(255*(img1/255)**2.05,dtype='uint8')
    display_output(gammaplt)
    return gammaplt


def maskimg():
    remove_output_Image()
    x, y, w, h = cv2.selectROI(img1)
    start = (x, y)
    end = (x + w, y + h)
    rect = (x, y, w, h)

    cv2.rectangle(img1, start, end, (0, 0, 255), 3)
    mask = np.zeros(img1.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img1, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask1 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    maskimage = image * mask1[:, :, np.newaxis]

    display_output(maskimage)
    return maskimage


def pencil():
    remove_output_Image()
    img1 = grayscale()
    img_invert = cv2.bitwise_not(img1)
    img_smoothing = cv2.GaussianBlur(img_invert, (25, 25), sigmaX=0, sigmaY=0)
    pencilimg = cv2.divide(img1, 255 - img_smoothing, scale=255)
    display_output(pencilimg)
    return pencilimg


def colpencil():
    remove_output_Image()
    img_invert = cv2.bitwise_not(img1)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    colpencilimg = cv2.divide(img1, 255 - img_smoothing, scale=255)
    display_output(colpencilimg)
    return colpencilimg


def cartoon():
    remove_output_Image()
    gray = grayscale()
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img1, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    display_output(cartoon)
    return cartoon

def watercolor():
    remove_output_Image()
    watercolor = cv2.stylization(img1, sigma_s=100, sigma_r=0.45)
    display_output(watercolor)
    return watercolor

def emboss():
    remove_output_Image()
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    emboss = cv2.filter2D(img1, kernel=kernel, ddepth=-1)
    emboss = cv2.cvtColor(emboss, cv2.COLOR_BGR2GRAY)
    emboss = 255 - emboss
    display_output(emboss)
    return emboss

def mytrans():
    from  Gradio_UI_Language_Tralnsaltor import my_translator
    my_translator()


root = Tk()
#tk.Toplevel()
root.title("V-Sense")
root.state('zoomed')


image = cv2.imread('C:/Users/admin/Python Projects/ImageStiching/Images/logo.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = Image.fromarray(image)
image1 = ImageTk.PhotoImage(image1)
panelE = Label(image=image1, borderwidth=5, relief="sunken")
panelE.image = image1
panelE.grid(row=0, column=0, rowspan=3, padx=2, pady=8)

l1= Label(root, text="Welcome to SamSan's V-Sense Platform",
           fg="white", bg="purple", width= 98, borderwidth=5, relief="groove",  font =('Verdana', 15))
l1.grid(row= 0, column= 1, columnspan= 6, padx=1, pady=10, sticky='nesw')


Upload_icon_path='C:/Users/admin/Python Projects/ImageStiching/Images/upload_icon.png'
image = cv2.imread(Upload_icon_path)
image = cv2.resize(image, (50, 50))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = Image.fromarray(image)
im = ImageTk.PhotoImage(image1)





#btn22= Button(root, text="", image=im, fg="white", bg="gray", compound= LEFT, command=uploadImage1,borderwidth=0)

btn22= Button(root, text="Upload_img1", fg="white", bg="gray", command=uploadImage1,font = fnt.Font(size = 10),width=28)
btn22.grid(row= 20, column= 1, sticky='nesw',columnspan= 1)


btn20= Button(root, text="Upload_img2", fg="white", bg="gray", command=uploadImage2,font = fnt.Font(size = 10),width=28)
btn20.grid(row= 20, column= 2,  sticky='nesw',columnspan= 1)

btn21= Button(root, text="Upload_img3", fg="white", bg="gray", command=uploadImage3,font = fnt.Font(size = 10),width=28)
btn21.grid(row= 20, column= 3,  sticky='nesw',columnspan= 1)

btn41= Button(root, text="Output Image", fg="white", bg="gray", command="",width=28)
btn41.grid(row= 20, column= 4,  sticky='nesw',columnspan= 1)

btn28= Button(root, text="Reset", fg="white", bg="black",  command= remove,font = fnt.Font(size = 10),width=27, height=1)
btn28.grid(row= 21, column= 2, padx=1, pady=5, sticky='nesw',columnspan= 1)



btn1= Button(root, text="Remove_Background", fg="white", bg="gray", command=Remove_Background,font = fnt.Font(size = 10))
btn1.grid(row= 13, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn2= Button(root, text="Change_background", fg="white", bg="gray", command=change_background,font = fnt.Font(size = 10))
btn2.grid(row= 14, column= 0, padx=1, pady=1, sticky='nesw')

btn10= Button(root, text="Change_background_video", fg="white", bg="gray", command=Changebackgroundvideo)
btn10.grid(row= 15, column= 0, padx=1, pady=1, sticky='nesw,',columnspan= 1)

btn3= Button(root, text="Remove_Rain", fg="white", bg="gray", command=Remove_Rain,font = fnt.Font(size = 10))
btn3.grid(row= 16, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn4= Button(root, text="Remove_Fog", fg="white", bg="gray", command=Remove_Fog,font = fnt.Font(size = 10))
btn4.grid(row=17, column= 0, padx=1, pady=1, sticky='nesw')

btn5= Button(root, text="Image_stitching", fg="white", bg="gray", command=stitch,font = fnt.Font(size = 10))
btn5.grid(row= 18, column= 0, padx=1, pady=1, sticky='nesw,',columnspan= 1)

btn6= Button(root, text="Face_Detection", fg="white", bg="gray", command=Face_Detection,font = fnt.Font(size = 10))
btn6.grid(row= 19, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn7= Button(root, text="Lane Departure", fg="white", bg="gray", command=lanedeparture,font = fnt.Font(size = 10))
btn7.grid(row= 20, column= 0, padx=1, pady=1, sticky='nesw')

btn8= Button(root, text="Object_Detection", fg="white", bg="gray", command=Object_Detection,font = fnt.Font(size = 10))
btn8.grid(row= 21, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn9= Button(root, text="Dms", fg="white", bg="gray", command = DMS_new,font = fnt.Font(size = 10))
btn9.grid(row= 22, column= 0, padx=1, pady=1, sticky='nesw')

# btn10= Button(root, text="Translator", fg="white", bg="blue", command=mytrans)
# btn10.grid(row= 4, column= 5, padx=1, pady=1, sticky='nesw,',columnspan= 1)




Changebackgroundvideo



btn21= Button(root, text="GrayScale", fg="white", bg="gray", command=grayscale,font = fnt.Font(size = 10))
btn21.grid(row= 6, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn22= Button(root, text="Contrast Enhancement", fg="white", bg="gray", command=histo,font = fnt.Font(size = 10))
btn22.grid(row= 7, column= 0, padx=1, pady=1, sticky='nesw')

btn24= Button(root, text="Sharpening", fg="white", bg="gray", command=sharp,font = fnt.Font(size = 10))
btn24.grid(row= 8, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn25= Button(root, text="Pencil Sketch", fg="white", bg="gray", command=pencil,font = fnt.Font(size = 10))
btn25.grid(row=9, column= 0, padx=1, pady=1, sticky='nesw')

btn26= Button(root, text="Water Color", fg="white", bg="gray", command=watercolor,font = fnt.Font(size = 10))
btn26.grid(row= 10, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn27= Button(root, text="Emboss", fg="white", bg="gray", command=emboss,font = fnt.Font(size = 10))
btn27.grid(row= 11, column= 0, padx=1, pady=1, sticky='nesw')

btn27= Button(root, text="", fg="gray", bg="gray", command='',font = fnt.Font(size = 10))
btn27.grid(row= 12, column= 0,  sticky='nesw')
#btn27.config(width=2, height=6)



# btn2= Button(root, text="INVERT COLOR", fg="white", bg="black", command=negative)
# btn2.grid(row= 7, column= 0, padx=1, pady=1, sticky='nesw')
#
# btn3= Button(root, text="RED ATTRIBUTES", fg="white", bg="red", command=redext)
# btn3.grid(row= 8, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)
#
# btn4= Button(root, text="GREEN ATTRIBUTES", fg="white", bg="green", command=greenext)
# btn4.grid(row= 9, column= 0, padx=1, pady=1, sticky='nesw')
#
# btn5= Button(root, text="BLUE ATTRIBUTES", fg="white", bg="blue", command=blueext)
# btn5.grid(row= 10, column= 0, padx=1, pady=1, sticky='nesw,',columnspan= 1)
#
# btn6= Button(root, text="BINARY", fg="white", bg="black", command=threshold)
# btn6.grid(row= 11, column= 0, padx=1, pady=1, sticky='nesw')
#
# btn7= Button(root, text="EDGE DETECTION", fg="white", bg="black", command=edge)
# btn7.grid(row= 12, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)
#
# btn8= Button(root, text="SKELETON", fg="white", bg="black", command=skeleton)
# btn8.grid(row= 13, column= 0, padx=1, pady=1, sticky='nesw')

# btn9= Button(root, text="POWER LAW TRANSFORMATION", fg="white", bg="purple", command=powerlawtrans)
# btn9.grid(row= 14, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

# btn12= Button(root, text="SMOOTHENING", fg="white", bg="purple", command=denoise)
# btn12.grid(row= 17, column= 0, padx=1, pady=1, sticky='nesw')

# btn13= Button(root, text="REMOVE BACKGROUND", fg="white", bg="purple", command=maskimg)
# btn13.grid(row= 18, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

# btn15= Button(root, text="COLOR PENCIL SKETCH", fg="white", bg="purple", command=colpencil)
# btn15.grid(row= 20, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

# btn16= Button(root, text="CARTOONIFY", fg="white", bg="purple", command=cartoon)
# btn16.grid(row= 21, column= 0, padx=1, pady=1, sticky='nesw')

# btn19= Button(root, text="EMBOSS image", fg="white", bg="purple", command=emboss)
# btn19.grid(row= 24, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

root.mainloop()