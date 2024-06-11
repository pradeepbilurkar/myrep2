import tkinter as tk
from tkinter import messagebox, ttk
from playsound import playsound
from tkinter import *

import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import pygame
import os
import pygame
from langdetect import detect
import pyttsx3
import time
pygame.init()
pygame.mixer.init()
import gradio as gr
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

css =" "

def locker():
    with gr.Blocks(css=css, theme=gr.themes.Glass()) as demo:  # theme=gr.themes.Glass()
        # with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            # with gr.Column():
            #       gr.Textbox ('SamSan Technolgies V-Sense Framework')
            with gr.Row():
                with gr.Column(scale=1, equal_height=False):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=.1):
                            img_logo = gr.Image("C:/Users/admin/Desktop/logo.png", equal_height=True)
                        with gr.Row(scale=1):
                            method = gr.Radio([["Remove Background"], "Change Background", 'Remove_Rain', 'Remove_Fog',
                                               'Image stitching', 'Face Detection', 'Lane Departure',
                                               'Object Detection'], label="method")
                        with gr.Row(scale=1):
                            outputs = gr.Image()

                    #     with gr.Row(scale=1):
                    # #f = gr.File(file_types=["image"], file_count="multiple")
                    #             method = gr.Radio([ "Remove Background", "Change Background",Remove_Rain,'Remove_Fog',"Image stitching",'Face Detection','Lane Departure'],label="method", info="Where did they go?")
                    with gr.Row(scale=1):
                        inputs1 = gr.Image()
                        inputs2 = gr.Image()
                        inputs3 = gr.Image()
                    # with gr.Row(scale=1):
                    # method = gr.Radio(["Image stitching", "Remove Background", "Change Background", 'Lane Departure', 'Face Detection','Remove_Fog'],label="method", info="Where did they go?")
                    # with gr.Column(scale=5):
                    #         outputs = gr.Image()
        with gr.Column():
            btn = gr.Button()

        # f.select(preview, f, i)
        btn.click(stitch, [method, inputs1, inputs2, inputs3], outputs)

    demo.launch()


def test():
    output1=3+4
    return output1

def detect_and_match_features(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    # keypoints3, descriptors3 = orb.detectAndCompute(img3, None)

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
    return blended_img.astype(np.uint8)


def call_rain_removal(img):
    lambda_1= 2e-2
    kappa = 2.0
    beta_max = 1e5

    S = img/ 256
    if S.ndim < 3:
        S = S[..., np.newaxis]

    N, M, D = S.shape

    beta = 2 *lambda_1

    psf = np.asarray([[-1, 1]])
    out_size = (N, M)
    otfx = psf2otf(psf, out_size)
    psf = np.asarray([[-1], [1]])
    otfy = psf2otf(psf, out_size)

    Normin1 = fft2(np.squeeze(S), axes=(0, 1))
    Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))
    if D > 1:
        Denormin2 = Denormin2[..., np.newaxis]
        Denormin2 = np.repeat(Denormin2, 3, axis=2)

    while beta < beta_max:
        Denormin = 1 + beta * Denormin2

        h = np.diff(S, axis=1)
        last_col = S[:, 0, :] - S[:, -1, :]
        last_col = last_col[:, np.newaxis, :]
        h = np.hstack([h, last_col])

        v = np.diff(S, axis=0)
        last_row = S[0, ...] - S[-1, ...]
        last_row = last_row[np.newaxis, ...]
        v = np.vstack([v, last_row])

        grad = np.square(h) + np.square(v)
        if D > 1:
            grad = np.sum(grad, axis=2)
            idx = grad < (lambda_1 / beta)
            idx = idx[..., np.newaxis]
            idx = np.repeat(idx, 3, axis=2)
        else:
            grad = np.sum(grad, axis=2)
            idx = grad < (lambda_1 / beta)

        h[idx] = 0
        v[idx] = 0

        h_diff = -np.diff(h, axis=1)
        first_col = h[:, -1, :] - h[:, 0, :]
        first_col = first_col[:, np.newaxis, :]
        h_diff = np.hstack([first_col, h_diff])

        v_diff = -np.diff(v, axis=0)
        first_row = v[-1, ...] - v[0, ...]
        first_row = first_row[np.newaxis, ...]
        v_diff = np.vstack([first_row, v_diff])

        Normin2 = h_diff + v_diff
        # Normin2 = beta * np.fft.fft2(Normin2, axes=(0, 1))
        Normin2 = beta * fft2(Normin2, axes=(0, 1))

        FS = np.divide(np.squeeze(Normin1) + np.squeeze(Normin2),
                       Denormin)
        # S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
        S = np.real(ifft2(FS, axes=(0, 1)))
        if False:
            S_new = S * 256
            S_new = S_new.astype(np.uint8)
            cv2.imshow('L0-Smooth', S_new)
            cv2.waitKey(0)

        if S.ndim < 3:
            S = S[..., np.newaxis]
        beta = beta * kappa
        S = np.squeeze(S)
        S = np.clip(S, 0, 1)
        S = S * 255
        S = S.astype(np.uint8)
        return S


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

def im_stitching(img1,img2):
    keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)
    H, mask = estimate_homography(keypoints1, keypoints2, matches)
    warped_img = warp_images(img2, img1, H)
    img1 = cv2.resize(img1, (warped_img.shape[1], warped_img.shape[0]))
    output_img = blend_images(warped_img, img1)
    # output_img = blend_images(img1,warped_img)
    #img1_original = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    #numpy_horizontal = np.hstack((img1_original,img2, output_img))
    cv2.imwrite('C:/Users/admin/Python Projects/ImageStiching/stitchedImage.jpg', output_img)
    return output_img


def preview(files, sd: gr.SelectData):
    return files[sd.index].name



from gradio.themes.base import Base


class Seafoam(Base):
    pass

seafoam = Seafoam()



#
# with gr.Blocks(css=css,theme=seafoam) as demo:  #theme=gr.themes.Glass()
#     gr.Markdown(
#         """
#         # Welcome to Exciting Demo of SamSan Digital
#         """)
#     with gr.Column(scale=1):
#           with gr.Row():
#             with gr.Column(scale=1,equal_height=False):
#                     # with gr.Column(scale=.1):
#                     #     img_logo = gr.Image("C:/Users/admin/Desktop/logo.png",equal_height=True)
#                     with gr.Row(scale=1):
#                         method = gr.Radio([ "Remove Background", "Change Background",'Remove_Rain','Remove_Fog',
#                                             'Image stitching','Face Detection','Lane Departure','Object Detection','DMS', 'Translator'],label="method")
#                     with gr.Row(scale=1):
#                         inputs1 = gr.Image()
#                         inputs2 = gr.Image()
#                         inputs3 = gr.Image()
#                         outputs = gr.Image()
#
#     with gr.Row():
#             #img_logo = gr.Image("C:/Users/admin/Desktop/logo.png", scale=0.005)
#             btn = gr.Button('Process',title="Hello 'Name' App",)
#     #with gr.Row():
#             #btn1 = gr.Button()
#
# #f.select(preview, f, i)
#     btn.click(stitch, [method, inputs1,inputs2,inputs3],outputs)
#
# demo.launch(height=700)






# with gr.Blocks() as demo:
#     # with gr.Row():
#     #     img_logo = gr.Image("C:/Users/admin/Desktop/logo.png", scale=1)
#         with gr.Column():
#             #f = gr.File(file_types=["image"], file_count="multiple")
#             method = gr.Radio(["Image stitching", "Remove Background", "Change Background", 'Lane Departure', 'Face Detection','Remove_Fog'],label="method", info="Where did they go?")
#             with gr.Row():
#                 inputs1 = gr.Image()
#                 inputs2 = gr.Image()
#                 inputs3= gr.Image()
#         with gr.Column():
#             outputs = gr.Image()
#             with gr.Column():
#                 btn = gr.Button()
#
# #f.select(preview, f, i)
#     btn.click(stitch, [method, inputs1,inputs2,inputs3],outputs)


def msg1():
    messagebox.showinfo('information', 'Please say something !!')

# Create the main application window
root = tk.Tk()
root.title("Language Translator")
root.geometry("500x150")
#root.geometry("600x850")
root.resizable(0, 0)
# input_text_label.pack(pady=5)
root.config(bg='#5FB691')


# Create a button to translate the text
photo = tk.PhotoImage(file='C:/Users/admin/Desktop/logo.png')
image_label = ttk.Label( root,image=photo, padding=5)
image_label.pack()
image_label.place(x=15,y=15)
# combos = ttk.Combobox(state="readonly", values=["English","Hindi", "Marathi", "Bangla", "Kannada"])  #alues=["en","hi", "mr", "bn", "kn"]
# combos.place(x=210, y=15)
combot = ttk.Combobox(state="readonly", values=["English","Hindi", "Marathi", "Bangla", "Kannada"])
combot.place(x=380, y=15)
combot.place(x=180, y=15)

# output_text_label = tk.Label(text="Source Language", compound='left')
# output_text_label.place(x=250,y=40)

output_text_label = tk.Label(text="Target Language", compound='left')
output_text_label.place(x=330,y=15)
#output_text_label.place(x=425,y=40)

exit_button = tk.Button(root, text="Exit", command=root.destroy,font=("Arial", 10), bg='#000', fg='#ff0',padx = 25, pady = 6)
exit_button.place(x=230, y=70)
translate_button = tk.Button(root, text="Speak", command=stitch,font=("Arial", 10),  bg='#000', fg='#ff0',padx = 20, pady = 5)
translate_button.place(x=330,y=70)
output_text = tk.Text(root, height=10, width=50)

root.mainloop()