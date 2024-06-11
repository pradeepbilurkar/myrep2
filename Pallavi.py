import PIL
from tkinter import *
from PIL import Image
from PIL import ImageTk
from PIL import ImageFilter
from tkinter import filedialog
import cv2
from skimage import filters
import skimage
import mahotas
import mahotas.demos
import numpy as np
from Gradio_laneDeparture import lanedeparture
from GradioFace_Identification import detect_face
from scipy.fftpack import fft2, ifft2
import cv2
from psf2otf import psf2otf
from Gradio_Rain_Smoothing import L0Smoothing
from Gradio_Object_detection import gradio_object
from Gradio_DMS import facial_processing
#from Gradio_Language_Translator import my_translator


global panelA,panelB,img1,img2,img3
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
    global panelA, panelB, img1
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image_original = image
    image = cv2.resize(image, (250, 350))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image1 = Image.fromarray(image)

    image1 = ImageTk.PhotoImage(image1)

    panelA = Label(image=image1, borderwidth=5, relief="sunken")
    panelA.image = image1
    panelA.grid(row=5, column=1, rowspan=15, padx=0, pady=0)

    img1 = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    return img1


def uploadImage2():
    global panelA, panelB, img2
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image_original = image
    image = cv2.resize(image, (250, 350))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image1 = Image.fromarray(image)

    image1 = ImageTk.PhotoImage(image1)

    panelA = Label(image=image1, borderwidth=5, relief="sunken")
    panelA.image = image1
    panelA.grid(row=5, column=2, rowspan=15, padx=0, pady=0)

    img2 = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    return img2


def uploadImage3():
    global panelA, panelB, img3
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    path = filedialog.askopenfilename(filetypes=f_types)
    image = cv2.imread(path)
    image_original = image
    image = cv2.resize(image, (250, 350))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image1 = Image.fromarray(image)

    image1 = ImageTk.PhotoImage(image1)

    panelA = Label(image=image1, borderwidth=5, relief="sunken")
    panelA.image = image1
    panelA.grid(row=5, column=3, rowspan=15, padx=0, pady=0)

    img3 = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    return img3


def display_output(output_image):
    global panelA, panelB
    import cv2
    output_image = cv2.resize(output_image, (250, 350))
    output_image1 = Image.fromarray(output_image)
    output_image1 = ImageTk.PhotoImage(output_image1)

    panelB = Label(image=output_image1, borderwidth=5, relief="sunken")
    panelB.image = output_image1
    panelB.grid(row=5, column=4, rowspan=15, padx=0, pady=0)


def remove():
    global panelA, panelB

    panelA = Label(image=None)
    panelA.grid(row=5, column=1, rowspan=15, padx=0, pady=0)
    panelA.image = None
    panelA = Label(image=None)
    panelA.grid(row=5, column=2, rowspan=15, padx=0, pady=0)
    panelA.image = None
    panelA.grid(row=5, column=3, rowspan=15, padx=0, pady=0)
    panelA.image = None
    panelB.grid(row=5, column=4, rowspan=15, padx=0, pady=0)
    panelB.image = None




def my_translator():
    from Gradio_Language_Translator import my_translator
    my_translator()

def change_background():
    import cv2
    # import numpy as np
    # from PIL import Image
    # from matplotlib import cm
    from rembg import remove

    # if img1 is None or img2 is None:
    #     return None, {'error': 'Error loading images. Check file paths.'}
    imgages = [img1, img2]
    imgages[0] = remove(imgages[0])

    1
    # if imgages[0].shape[:2] != imgages[1].shape[:2]:
    #      img2_pil = Image.fromarray(cv2.cvtColor(imgages[1], cv2.COLOR_BGR2RGB))
    #      img2_resized = img2_pil.resize((imgages[0].shape[1], imgages[0].shape[0]), Image.BICUBIC)
    #      img2 = cv2.cvtColor(np.array(img2_resized), cv2.COLOR_RGB2BGR)

    imgages[0] = cv2.cvtColor(imgages[0], cv2.COLOR_BGR2RGB)
    imgages[1] = cv2.cvtColor(imgages[1], cv2.COLOR_BGR2RGB)
    gray_img1 = cv2.cvtColor(imgages[0], cv2.COLOR_RGB2GRAY)
    _, alpha = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY)
    alpha_inv = cv2.bitwise_not(alpha)
    img1_region = cv2.bitwise_and(imgages[0], imgages[0], mask=alpha)
    img2_region = cv2.bitwise_and(imgages[1], imgages[1], mask=alpha_inv)
    blended_image = cv2.addWeighted(img1_region, 1, img2_region, 1, 0)
    result = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
         # cv2.imshow(imgages[1])
         # cv2.waitkey(0)
         # result = cv2.resize(result, (400, 300))
    display_output(result)
    return result


def stitch():
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
        # cv2.imshow('stitched_image',stitched_image)
        # cv2.waitKey(0)
        #stitched_image = cv2.resize(stitched_image, (400,300))        # cv2.imshow('A',stitched_image)
        # cv2.waitKey(0)
        display_output(stitched_image)


        return stitched_image
    else:
        print("Image stitching failed!")
        return None

def Remove_Background():
    from rembg import remove
    import numpy as np
    import cv2
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
    import image_dehazer
    HazeCorrectedImg, haze_map = image_dehazer.remove_haze(img1, showHazeTransmissionMap=False)
    #HazeCorrectedImg = L0Smoothing(HazeCorrectedImg)
    display_output(HazeCorrectedImg)
    return HazeCorrectedImg

def Remove_Rain():
    out_image = L0Smoothing(img1)
    # import image_dehazer
    # HazeCorrectedImg, haze_map = image_dehazer.remove_haze(out_image, showHazeTransmissionMap=False)
    # return HazeCorrectedImg
    display_output(out_image)
    return out_image

def Object_Detection():
    gradio_object()


def Face_Detection():
    detect_face()




def grayscale():
    grayimg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    display_output(grayimg)
    # grayimg1 = Image.fromarray(grayimg)
    #
    # grayimg1 = ImageTk.PhotoImage(grayimg1)
    #
    # panelB = Label(image=grayimg1, borderwidth=1, relief="sunken")
    # panelB.image = grayimg1
    # panelB.grid(row=10, column=4, rowspan=1, padx=0, pady=0)

    return grayimg


def negative():
    neg = 255 - img1

    display_output(neg)
    # neg1 = Image.fromarray(neg)
    #
    # neg1 = ImageTk.PhotoImage(neg1)
    #
    # panelB = Label(image=neg1, borderwidth=1, relief="sunken")
    # panelB.image = neg1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return neg


def threshold():
    image = grayscale()

    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    display_output(thresh)
    # thresh1 = Image.fromarray(thresh)
    #
    # thresh1 = ImageTk.PhotoImage(thresh1)
    #
    # panelB = Label(image=thresh1, borderwidth=5, relief="sunken")
    # panelB.image = thresh1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return thresh


def redext():
    row, col, plane = img1.shape

    red = np.zeros((row, col, plane), np.uint8)
    red[:, :, 0] = img1[:, :, 0]

    display_output(red)
    # red1 = Image.fromarray(red)
    #
    # red1 = ImageTk.PhotoImage(red1)
    #
    # panelB = Label(image=red1, borderwidth=5, relief="sunken")
    # panelB.image = red1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return red


def greenext():
    row, col, plane = img1.shape

    green = np.zeros((row, col, plane), np.uint8)
    green[:, :, 1] = img1[:, :, 1]

    display_output(green)
    # green1 = Image.fromarray(green)
    #
    # green1 = ImageTk.PhotoImage(green1)
    #
    # panelB = Label(image=green1, borderwidth=5, relief="sunken")
    # panelB.image = green1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return green


def blueext():
    row, col, plane = img1.shape

    blue = np.zeros((row, col, plane), np.uint8)
    blue[:, :, 2] = img1[:, :, 2]

    display_output(blue)
    # blue1 = Image.fromarray(blue)
    #
    # blue1 = ImageTk.PhotoImage(blue1)
    #
    # panelB = Label(image=blue1, borderwidth=5, relief="sunken")
    # panelB.image = blue1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return blue


def edge():
    img1 = threshold()

    edged = cv2.Canny(img1, 50, 100)

    display_output(edged)
    # edged1 = Image.fromarray(edged)
    #
    # edged1 = ImageTk.PhotoImage(edged1)
    #
    # panelB = Label(image=edged1, borderwidth=5, relief="sunken")
    # panelB.image = edged1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return edged


def skeleton():
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
    # skel1 = Image.fromarray(skel)
    #
    # skel1 = ImageTk.PhotoImage(skel1)
    #
    # panelB = Label(image=skel1, borderwidth=5, relief="sunken")
    # panelB.image = skel1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return skel


def denoise():
    denoise = cv2.fastNlMeansDenoisingColored(img1, None, 5, 5, 7, 21)

    display_output(denoise)
    # denoise1 = Image.fromarray(denoise)
    #
    # denoise1 = ImageTk.PhotoImage(denoise1)
    #
    # panelB = Label(image=denoise1, borderwidth=5, relief="sunken")
    # panelB.image = denoise1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return denoise


def sharp():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img1, ddepth=-1, kernel=kernel)

    display_output(sharpened)

    # sharpened1 = Image.fromarray(sharpened)
    #
    # sharpened1 = ImageTk.PhotoImage(sharpened1)
    #
    # panelB = Label(image=sharpened1, borderwidth=5, relief="sunken")
    # panelB.image = sharpened1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return sharpened


def histo():
    histogram = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    histogram[:, :, 0] = cv2.equalizeHist(histogram[:, :, 0])
    histogram = cv2.cvtColor(histogram, cv2.COLOR_YUV2BGR)

    display_output(histogram)
    # histogram1 = Image.fromarray(histogram)
    #
    # histogram1 = ImageTk.PhotoImage(histogram1)
    #
    # panelB = Label(image=histogram1, borderwidth=5, relief="sunken")
    # panelB.image = histogram1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return histogram


def powerlawtrans():
    gammaplt = np.array(255*(img1/255)**2.05,dtype='uint8')

    display_output(gammaplt)
    # gammaplt1 = Image.fromarray(gammaplt)
    #
    # gammaplt1= ImageTk.PhotoImage(gammaplt1)
    #
    # panelB = Label(image=gammaplt1, borderwidth=5, relief="sunken")
    # panelB.image = gammaplt1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return gammaplt


def maskimg():
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
    # maskimage1 = Image.fromarray(maskimage)
    #
    # maskimage1 = ImageTk.PhotoImage(maskimage1)
    #
    # panelB = Label(image=maskimage1, borderwidth=5, relief="sunken")
    # panelB.image = maskimage1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return maskimage


def pencil():
    img1 = grayscale()
    img_invert = cv2.bitwise_not(img1)
    img_smoothing = cv2.GaussianBlur(img_invert, (25, 25), sigmaX=0, sigmaY=0)

    pencilimg = cv2.divide(img1, 255 - img_smoothing, scale=255)

    display_output(pencilimg)
    # pencilimg1 = Image.fromarray(pencilimg)
    #
    # pencilimg1 = ImageTk.PhotoImage(pencilimg1)
    #
    # panelB = Label(image=pencilimg1, borderwidth=5, relief="sunken")
    # panelB.image = pencilimg1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return pencilimg


def colpencil():
    img_invert = cv2.bitwise_not(img1)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)

    colpencilimg = cv2.divide(img1, 255 - img_smoothing, scale=255)

    display_output(colpencilimg)
    # colpencilimg1 = Image.fromarray(colpencilimg)
    #
    # colpencilimg1 = ImageTk.PhotoImage(colpencilimg1)
    #
    # panelB = Label(image=colpencilimg1, borderwidth=5, relief="sunken")
    # panelB.image = colpencilimg1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return colpencilimg


def cartoon():
    gray = grayscale()
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(img1, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    display_output(cartoon)
    # cartoon1 = Image.fromarray(cartoon)
    #
    # cartoon1 = ImageTk.PhotoImage(cartoon1)
    #
    # panelB = Label(image=cartoon1, borderwidth=5, relief="sunken")
    # panelB.image = cartoon1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)

    return cartoon


def watercolor():
    watercolor = cv2.stylization(img1, sigma_s=100, sigma_r=0.45)

    display_output(watercolor)
    # watercolor1 = Image.fromarray(watercolor)
    #
    # watercolor1 = ImageTk.PhotoImage(watercolor1)
    #
    # panelB = Label(image=watercolor1, borderwidth=5, relief="sunken")
    # panelB.image = watercolor1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return watercolor


def emboss():
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    emboss = cv2.filter2D(img1, kernel=kernel, ddepth=-1)
    emboss = cv2.cvtColor(emboss, cv2.COLOR_BGR2GRAY)
    emboss = 255 - emboss

    display_output(emboss)
    # emboss1 = Image.fromarray(emboss)
    #
    # emboss1 = ImageTk.PhotoImage(emboss1)
    #
    # panelB = Label(image=emboss1, borderwidth=5, relief="sunken")
    # panelB.image = emboss1
    # panelB.grid(row=8, column=4, rowspan=1, columnspan=1, padx=20, pady=20)


    return emboss



root = Tk()
root.title("IMAGE PROCESSING")

l1= Label(root, text="CLICK THE BUTTONS TO PERFORM THE FUNCTIONALITIES MENTIONED",
           fg="white", bg="purple", width= 98, borderwidth=5, relief="groove",  font =('Verdana', 15))
l1.grid(row= 0, column= 0, columnspan= 6, padx=20, pady=20, sticky='nesw')

btn22= Button(root, text="Upload_img1", fg="black", bg="lavender", command=uploadImage1)
btn22.grid(row= 20, column= 1, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn20= Button(root, text="Upload_img2", fg="black", bg="lavender", command=uploadImage2)
btn20.grid(row= 20, column= 2, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn21= Button(root, text="Upload_img3", fg="black", bg="lavender", command=uploadImage3)
btn21.grid(row= 20, column= 3, padx=1, pady=1, sticky='nesw',columnspan= 1)



# btn1= Button(root, text="Change_Background", fg="white", bg="snow4", command=my_translator)
# btn1.grid(row= 1, column= 3, padx=1, pady=1, sticky='nesw',columnspan= 1)
# #



btn1= Button(root, text="Remove_Background", fg="white", bg="snow4", command=Remove_Background)
btn1.grid(row= 3, column= 1, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn2= Button(root, text="change_background", fg="white", bg="black", command=change_background)
btn2.grid(row= 3, column= 2, padx=1, pady=1, sticky='nesw')

btn3= Button(root, text="Remove_Rain", fg="white", bg="red", command=Remove_Rain)
btn3.grid(row= 3, column= 3, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn4= Button(root, text="Remove_Fog", fg="white", bg="green", command=Remove_Fog)
btn4.grid(row= 3, column= 4, padx=1, pady=1, sticky='nesw')

btn5= Button(root, text="Image_stitching", fg="white", bg="blue", command=stitch)
btn5.grid(row= 3, column= 5, padx=1, pady=1, sticky='nesw,',columnspan= 1)

btn6= Button(root, text="Face_Detection", fg="white", bg="snow4", command=Face_Detection)
btn6.grid(row= 4, column= 1, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn7= Button(root, text="Lane Departure", fg="white", bg="black", command=lanedeparture)
btn7.grid(row= 4, column= 2, padx=1, pady=1, sticky='nesw')

btn8= Button(root, text="Object_Detection", fg="white", bg="red", command=Object_Detection)
btn8.grid(row= 4, column= 3, padx=1, pady=1, sticky='nesw',columnspan= 1)

# btn4= Button(root, text="DMS, fg="white", bg="green", command = facial_processing)
# btn4.grid(row= 4, column= 4, padx=1, pady=1, sticky='nesw')
#
# btn5= Button(root, text="BLUE ATTRIBUTES", fg="white", bg="blue", command=Translator)
# btn5.grid(row= 4, column= 5, padx=1, pady=1, sticky='nesw,',columnspan= 1)




#

btn1= Button(root, text="GRAYSCALE", fg="white", bg="snow4", command=grayscale)
btn1.grid(row= 6, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn2= Button(root, text="INVERT COLOR", fg="white", bg="black", command=negative)
btn2.grid(row= 7, column= 0, padx=1, pady=1, sticky='nesw')

btn3= Button(root, text="RED ATTRIBUTES", fg="white", bg="red", command=redext)
btn3.grid(row= 8, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn4= Button(root, text="GREEN ATTRIBUTES", fg="white", bg="green", command=greenext)
btn4.grid(row= 9, column= 0, padx=1, pady=1, sticky='nesw')

btn5= Button(root, text="BLUE ATTRIBUTES", fg="white", bg="blue", command=blueext)
btn5.grid(row= 10, column= 0, padx=1, pady=1, sticky='nesw,',columnspan= 1)

btn6= Button(root, text="BINARY", fg="white", bg="black", command=threshold)
btn6.grid(row= 11, column= 0, padx=1, pady=1, sticky='nesw')

btn7= Button(root, text="EDGE DETECTION", fg="white", bg="black", command=edge)
btn7.grid(row= 12, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn8= Button(root, text="SKELETON", fg="white", bg="black", command=skeleton)
btn8.grid(row= 13, column= 0, padx=1, pady=1, sticky='nesw')

btn9= Button(root, text="POWER LAW TRANSFORMATION", fg="white", bg="purple", command=powerlawtrans)
btn9.grid(row= 14, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn10= Button(root, text="CONTRAST ENHANCEMENT", fg="white", bg="purple", command=histo)
btn10.grid(row= 15, column= 0, padx=1, pady=1, sticky='nesw')

btn11= Button(root, text="SHARPENING", fg="white", bg="purple", command=sharp)
btn11.grid(row= 16, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn12= Button(root, text="SMOOTHENING", fg="white", bg="purple", command=denoise)
btn12.grid(row= 17, column= 0, padx=1, pady=1, sticky='nesw')

btn13= Button(root, text="REMOVE BACKGROUND", fg="white", bg="purple", command=maskimg)
btn13.grid(row= 18, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn14= Button(root, text="PENCIL SKETCH", fg="white", bg="purple", command=pencil)
btn14.grid(row= 19, column= 0, padx=1, pady=1, sticky='nesw')

btn15= Button(root, text="COLOR PENCIL SKETCH", fg="white", bg="purple", command=colpencil)
btn15.grid(row= 20, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn16= Button(root, text="CARTOONIFY", fg="white", bg="purple", command=cartoon)
btn16.grid(row= 21, column= 0, padx=1, pady=1, sticky='nesw')

btn17= Button(root, text="WATERCOLOR", fg="white", bg="purple", command=watercolor)
btn17.grid(row= 22, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn18= Button(root, text="EMBOSS", fg="white", bg="purple", command=emboss)
btn18.grid(row= 23, column= 0, padx=1, pady=1, sticky='nesw')

btn19= Button(root, text="EMBOSS image", fg="white", bg="purple", command=emboss)
btn19.grid(row= 24, column= 0, padx=1, pady=1, sticky='nesw',columnspan= 1)

btn19= Button(root, text="Clear All", fg="white", bg="black",  command= remove)
btn19.grid(row= 21, column= 2, padx=1, pady=1, sticky='nesw',columnspan= 1)



root.mainloop()