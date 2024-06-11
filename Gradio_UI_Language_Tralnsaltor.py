import gradio as gr

import numpy as np
from scipy.fftpack import fft2, ifft2
from psf2otf import psf2otf



css =" "

from gradio.themes.base import Base


class Seafoam(Base):
    pass

seafoam = Seafoam()
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import pygame
import os
import pygame
from langdetect import detect
import pyttsx3
import time
# pygame.init()
# pygame.mixer.init()
def recognize_speech_from_microphone():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as source
    with sr.Microphone() as source:
        #print("Please say something...")
        #msg = Message(root, text="A computer science portal for geeks")

        #msg.pack()
        # Adjust for ambient noise and record audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            src_lan=(detect(text))
            print(detect(text))
            print(f"Recognized: {text}")
            return text, src_lan
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    return None


def translate_text(text,src_lan,combotgt):
    # Initialize the translator
    #combosrc = combos.get()
    #combotgt = combot.get()
    #combosrc = lang_reco(combosrc)
    #combotgt = lang_reco(language)
    # Translate the recognized text to Hindi (or any other Indian language)
    target_language = combotgt  # 'hi'  # 'hi' for Hindi, 'ta' for Tamil, 'te' for Telugu, etc.
    #source_lang = combosrc
    translator = Translator()
    try:
        # Translate the text
        text_to_translate = translator.translate(text,  dest=combotgt,src=src_lan,)
        translated = translator.translate(text, dest=target_language)
        print(f"Translated to {target_language}: {translated.text}")
        return translated.text
    except Exception as e:
        print(f"Translation failed; {e}")
    return None

def lang_reco(combotgt):
    match combotgt:
        case "English":
            combotgt="en"
            return combotgt
        case "Hindi":
            combotgt = "hi"
            return combotgt
        case "Marathi":
            combotgt = "mr"
            return combotgt
        case "Bangla":
           combotgt = "bn"
           return combotgt
        case "Kannada":
            combotgt = "kn"
            return combotgt

def my_translator(language):

    from gtts import gTTS

    import os
    import pygame
    from langdetect import detect
    import pyttsx3
    import time
    # Recognize speech from the microphone
    try:
        os.remove('recorded_audio.mp3')
    except:
        1
    pygame.mixer.quit()

    recognized_text = recognize_speech_from_microphone()

    if recognized_text:

        combotgt=lang_reco(language)
        src_lan = recognized_text[1]
        # Translate the recognized text to Hindi (or any other Indian language)
        target_language = combotgt # 'hi'  # 'hi' for Hindi, 'ta' for Tamil, 'te' for Telugu, etc.
        #source_lang=combosrc
        text=translate_text(recognized_text, src_lan,combotgt)
        tts = gTTS(text=text, lang=target_language, slow=False)
        tts.save("recorded_audio.mp3")
        # engine = pyttsx3.init()
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     print("Voice:")
        #     print(" - ID: %s" % voice.id)
        #     print(" - Name: %s" % voice.name)
        #     print(" - Languages: %s" % voice.languages)
        #     print(" - Gender: %s" % voice.gender)
        #     print(" - Age: %s" % voice.age)
        # engine.say(text)
        # engine.runAndWait()
        1
        #playsound('captured_voice.mp3')
        # audio_file = "output.mp3"
        # # Play the converted file using pygame
        pygame.init()
        pygame.mixer.init()

        pygame.mixer.music.load("recorded_audio.mp3")
        pygame.mixer.music.play()

        #
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

            # Remove the audio file after playback
        pygame.mixer.quit()
        os.remove('recorded_audio.mp3')



with gr.Blocks(css=css,theme=seafoam) as demo:  #theme=gr.themes.Glass()
    gr.Markdown(
        "SamSan MultiLingual Translator"
      )
    with gr.Column(scale=1):
          with gr.Row():
            with gr.Column(scale=1,equal_height=False):
                    # with gr.Column(scale=.1):
                    #     img_logo = gr.Image("C:/Users/admin/Desktop/logo.png",equal_height=True)
                    with gr.Row(scale=1):
                        language=gr.Dropdown(
                            ["English", "Marathi", "Kannada","Bangla","Hindi"], label="Language")


    with gr.Row():
            #img_logo = gr.Image("C:/Users/admin/Desktop/logo.png", scale=0.005)
            btn = gr.Button('Process',title="Hello 'Name' App",)

#f.select(preview, f, i)
    btn.click(my_translator,inputs=[language])

demo.launch(height=700)






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
