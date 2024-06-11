from tkinter import *
from PIL import ImageTk, Image
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np


root = Tk()

# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

# Capture from camera
isclosed=0
countframe = 0
missed = 0
#video_capture = cv2.VideoCapture('C:/Users/admin/Python Projects/MyTestChatGpt/VID_20230905_175819708_HL.mp4')
cap = cv2.VideoCapture('C:/Users/admin/Python Projects/\LaneDeparture_Working/MyCar_test1video.mp4')

# function for video streaming
def video_stream():
    missed = 0
    countframe=0
   #while (True):
    _, image = cap.read()
    image = cv2.resize(image, (700, 900))
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    # cv2.imshow('image', image)
    # Step 2: Reading the Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Converting to Grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Gaussian Blur
    # edges = cv2.Canny(blur, 50, 150)  # working
    edges = cv2.Canny(blur, 40, 165)

    # Step 5: Canny Edge Detection
    height, width = image.shape[:2]
    #   roi_vertices = [(0, height/1.9), (width-width*.5, height/3), (width-width*0.2, height-height*.6)]  # working to some extent
    roi_vertices = [(0, height / 1.9), (width / 2, height / 3), (width - width * 0.2, height - height * .6)]
    mask_color = 255
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Step 6: Region of Interest
    # lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25) # original
    # lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi / 60, threshold=100, minLineLength=25, maxLineGap=30) # working to some extent
    lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi / 60, threshold=90, minLineLength=25, maxLineGap=45)

    # Step 7: Hough Transform
    line_image = np.zeros_like(image)
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  original
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Step 8: Drawing the Lines
        # final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0) original
        final_image = cv2.addWeighted(image, .9, line_image, 2, 1)
        cv2image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGBA)
        # Step 9: Overlaying the Lines on the Original Image
        # cv2.imshow('image',final_image)
        countframe = countframe + 1
    except:
        missed = missed + 1
        countframe = countframe + 1
        # print(countframe)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)

video_stream()
root.mainloop()








#
# root = Tk()
# # Create a frame
# app = Frame(root, bg="white")
# app.grid()
# # Create a label in the frame
# lmain = Label(app)
# lmain.grid()
#
# # Capture from camera
# cap = cv2.VideoCapture(0)
#
# # function for video streaming
# def video_stream():
#     _, frame = cap.read()
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(1, video_stream)
#
# video_stream()
# root.mainloop()