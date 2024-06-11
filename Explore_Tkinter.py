import PIL.Image
from tkinter import *
from PIL import Image
from PIL import ImageTk
from PIL import ImageFilter
from tkinter import filedialog
import os
import cv2
import tkinter as tk
from skimage import filters
import skimage
# import mahotas
# import mahotas.demos
import numpy as np
import PIL.Image
import tkinter as tk
from tkinter import *
import tkinter as tk
from tkinter import *
from Gradio_laneDeparture import lanedeparture
from GradioFace_Identification import detect_face
import numpy as np
from scipy.fftpack import fft2, ifft2
import cv2
from psf2otf import psf2otf
from Gradio_Rain_Smoothing import L0Smoothing
from Gradio_Object_detection import gradio_object

from rembg import remove
#from Test_mainFunction import removeBG
from PIL import Image
#from backgroundremover.bg import remove
from Gradio_DMS import facial_processing


def stitch(method,img1, img2,img3):
    import cv2
    if method =='Image stitching':
        stitcher = cv2.Stitcher_create()
        if img3 is None:
           imgages=[img1, img2]
        else:
            imgages = [img1, img2, img3]
        status, stitched_image = stitcher.stitch(imgages)

        if status == cv2.Stitcher_OK:
            cv2.imwrite('C:/Users/admin/Python Projects/ImageStiching/OutputImages/stitchedImage.jpg', stitched_image)
            # cv2.imshow('stitched_image',stitched_image)
            # cv2.waitKey(0)
            #stitched_image = cv2.resize(stitched_image, (400,300))
            return stitched_image
        else:
            print("Image stitching failed!")
            return None
        #stitched_image=im_stitching (img1,img2)
        #return stitched_image

    if method == 'Remove Background':
        from rembg import remove
        import numpy as np
        import cv2
        if (img2 and img3) is None:
            img1=remove(img1)
            return img1
        if (img1 and img3) is None:
            img2 = remove(img2)
            return img2
        if (img1 and img2) is None:
            img4= remove(img3)
            return img4

    if method == 'Change Background':
        import cv2
        import numpy as np
        from PIL import Image
        from  matplotlib import cm
        from rembg import remove

        if img1 is None or img2 is None:
            return None, {'error': 'Error loading images. Check file paths.'}
        img1 = remove(img1)
        if img1.shape[:2] != img2.shape[:2]:
            img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            img2_resized = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.BICUBIC)
            img2 = cv2.cvtColor(np.array(img2_resized), cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY)
        alpha_inv = cv2.bitwise_not(alpha)
        img1_region = cv2.bitwise_and(img1, img1, mask=alpha)
        img2_region = cv2.bitwise_and(img2, img2, mask=alpha_inv)
        blended_image = cv2.addWeighted(img1_region, 1, img2_region, 1, 0)
        result = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow(img1)
        # cv2.waitkey(0)
        result = cv2.resize(result, (400, 300))
        return result

    if method == 'Lane Departure':
        lanedeparture()
    if method == 'Face Detection':
        detect_face()
    if method== 'Remove_Fog':
        import image_dehazer
        HazeCorrectedImg, haze_map = image_dehazer.remove_haze(img1, showHazeTransmissionMap=False)
        #HazeCorrectedImg = L0Smoothing(HazeCorrectedImg)
        return HazeCorrectedImg
    if method == "Remove_Rain":
        out_image=L0Smoothing(img1)
        # import image_dehazer
        # HazeCorrectedImg, haze_map = image_dehazer.remove_haze(out_image, showHazeTransmissionMap=False)
        #return HazeCorrectedImg
        return out_image
    if method == "Object Detection":
        gradio_object()
    if method == "DMS":
        facial_processing()
    if method=='Translator':
        from Gradio_Language_Translator import my_translator
        my_translator()





def showimage():
    # filename=filedialog.askopenfilename(initialdir=os.getcwd(),
    #                                     title="Select image file", filetypes=(("PNG file", "*.png"),
    #                                                                           ("JPG file", "*.jpg"),
    #                                                                           ("JPEG file", "*.jpeg"),
    #                                                                           ("ALL file", "*.txt")))
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                         title="Select image file")
    importedimage = Image.open(filename)
    importedimage = ImageTk.PhotoImage(importedimage)
    lbl.configure(image=importedimage, width=380, height=320)
    lbl.image=importedimage

def showimage1(self):
    file_path = filedialog.askopenfilename()
    if file_path:
        self.image = cv2.imread(file_path)
        self.display_image()

def sel():
   selection = "You selected the option " + str(var.get())
   label.config(text = selection)

window = Tk()
window.title("Pictures transformer")
window.geometry("900x800+100+100")
window.configure(bg="#e2f9b8")
method = StringVar()
from PIL import Image, ImageTk
image=cv2.imread("C:/Users/admin/Desktop/logo.png")
color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_coverted)
#pil_image.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = ImageTk.PhotoImage(image=Image.fromarray(image))
Label(image=image, bg="#fff").place(x=10, y=10)
Label(text="Pictures transformer", font="arial 30 bold", fg="#313715", bg="#e2f9b8").place(x=120, y=50)

# selected image
img1 = Frame(width=300, height=315, bg="#d6dee5")
img1.place(x=30, y=130)
f = Frame(img1, bg="red", width=280, height=260)
f.place(x=10, y=10)
lbl = Label(f, bg="red")
lbl.place(x=0, y=0)


Button(img1, text="Select image", width=12, height=2, font="arial 10" , command=showimage).place(x=90, y=270)

img2 = Frame(width=300, height=315,bg="#d6dee5")
img2.place(x=350, y=130)
f1 = Frame(img2, bg="red", width=280, height=260)
f1.place(x=10, y=10)
lbl = Label(f1, bg="red")
lbl.place(x=0, y=0)
Button(img2, text="Select image", width=12, height=2, font="arial 10 ", command=showimage).place(x=90, y=270)

img3 = Frame(width=300, height=315,bg="#d6dee5")
img3.place(x=670, y=130)
f1 = Frame(img3, bg="red", width=280, height=260)
f1.place(x=10, y=10)
lbl = Label(f1, bg="red")
lbl.place(x=0, y=0)
Button(img3, text="Select image", width=12, height=2, font="arial 10 ", command=showimage).place(x=90, y=270)

img4 = Frame(width=300, height=315,bg="#d6dee5")
img4.place(x=985, y=130)
f1 = Frame(img4, bg="red", width=280, height=260)
f1.place(x=10, y=10)
lbl = Label(f1, bg="red")
lbl.place(x=0, y=0)

var = IntVar()
R1 = Radiobutton(window, text="Remove Background", variable=method, value=1, command=sel)
R1.place(x=90, y=480)
#R1.pack( anchor = W )
R2 = Radiobutton(window, text="Change Background", variable=method, value=2, command=sel)
R2.place(x=240, y=480)
#R2.pack( anchor = W )
R3 = Radiobutton(window, text="Remove_Rain", variable=method, value=3, command=sel)
R3.place(x=390, y=480)
R4 = Radiobutton(window, text="Remove_Fog", variable=method, value=4, command=sel)
R4.place(x=510, y=480)
R5 = Radiobutton(window, text="Image stitching", variable=method, value=5, command=sel)
R5.place(x=660, y=480)
R6 = Radiobutton(window, text="Face Detection", variable=method, value=6, command=sel)
R6.place(x=810, y=480)
R7 = Radiobutton(window, text="Lane Departure", variable=method, value=7, command=sel)
R7.place(x=960, y=480)
R8 = Radiobutton(window, text="Object Detection", variable=method, value=8, command=sel)
R8.place(x=1110, y=480)
#R3.pack( anchor = W)
label = Label(window)
label.pack()

btn = Button(window, text='Process', command=stitch(method,img1, img2,img3),font="arial 14 ").place(x=600, y=550)

#Button(processedimage, text="Select image", width=12, height=2, font="arial 14 ", command=showimage).place(x=10, y=340)
window.mainloop()
# from tkinter import filedialog
# import os
#

#
