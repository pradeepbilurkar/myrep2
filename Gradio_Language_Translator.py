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
# pygame.init()
# pygame.mixer.init()
def recognize_speech_from_microphone():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as source
    with sr.Microphone() as source:
        #print("Please say something...")
        msg = Message(root, text="A computer science portal for geeks")

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


def translate_text(text,src_lan):
    # Initialize the translator
    #combosrc = combos.get()
    combotgt = combot.get()
    #combosrc = lang_reco(combosrc)
    combotgt = lang_reco(combotgt)
    # Translate the recognized text to Hindi (or any other Indian language)
    target_language = combotgt  # 'hi'  # 'hi' for Hindi, 'ta' for Tamil, 'te' for Telugu, etc.
    #source_lang = combosrc
    translator = Translator()
    try:
        # Translate the text
        text_to_translate = translator.translate(text,  dest=target_language,src=src_lan,)
        translated = translator.translate(text, dest=target_language)
        print(f"Translated to {target_language}: {translated.text}")
        return translated.text
    except Exception as e:
        print(f"Translation failed; {e}")
    return None

def lang_reco(combosrc):
    match combosrc:
        case "English":
           combosrc="en"
           return combosrc
        case "Hindi":
            combosrc = "hi"
            return combosrc
        case "Marathi":
            combosrc = "mr"
            return combosrc
        case "Bangla":
            combosrc = "bn"
            return combosrc
        case "Kannada":
            combosrc = "kn"
            return combosrc
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

def my_translator():

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
        #combosrc = combos.get()
        combotgt = combot.get()
        #combosrc=lang_reco(combosrc)
        combotgt=lang_reco(combotgt)
        src_lan = recognized_text[1]
        # Translate the recognized text to Hindi (or any other Indian language)
        target_language = combotgt # 'hi'  # 'hi' for Hindi, 'ta' for Tamil, 'te' for Telugu, etc.
        #source_lang=combosrc
        text=translate_text(recognized_text, src_lan)
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

    root = tk.Tk()
    root.title("Language Translator")
    root.geometry("500x150")
    # root.geometry("600x850")
    root.resizable(0, 0)
    # input_text_label.pack(pady=5)
    root.config(bg='#5FB691')

    # Create a button to translate the text
    photo = tk.PhotoImage(file='C:/Users/admin/Desktop/logo.png')
    image_label = ttk.Label(root, image=photo, padding=5)
    image_label.pack()
    image_label.place(x=15, y=15)
    # combos = ttk.Combobox(state="readonly", values=["English","Hindi", "Marathi", "Bangla", "Kannada"])  #alues=["en","hi", "mr", "bn", "kn"]
    # combos.place(x=210, y=15)
    combot = ttk.Combobox(state="readonly", values=["English", "Hindi", "Marathi", "Bangla", "Kannada"])
    combot.place(x=380, y=15)
    combot.place(x=180, y=15)

    # output_text_label = tk.Label(text="Source Language", compound='left')
    # output_text_label.place(x=250,y=40)

    output_text_label = tk.Label(text="Target Language", compound='left')
    output_text_label.place(x=330, y=15)
    # output_text_label.place(x=425,y=40)

    exit_button = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 10), bg='#000', fg='#ff0', padx=25,
                            pady=6)
    exit_button.place(x=230, y=70)
    translate_button = tk.Button(root, text="Speak", command=my_translator, font=("Arial", 10), bg='#000', fg='#ff0',
                                 padx=20, pady=5)
    translate_button.place(x=330, y=70)
    output_text = tk.Text(root, height=10, width=50)

    root.mainloop()
def translate_tex1():
    try:
        # Get the text from the input field
        text_to_translate = input_text.get("1.0", tk.END).strip()

        if not text_to_translate:
            messagebox.showwarning("Input Error", "Please enter text to translate")
            return

        # Create a Translator object
        translator = Translator()

        # Translate the text to Hindi
        translated = translator.translate(text_to_translate, dest=combotgt)

        # Display the translated text
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, translated.text)
    except Exception as e:
        messagebox.showerror("Translation Error", str(e))



def msg1():
    messagebox.showinfo('information', 'Please say something !!')

# Create the main application window
root = Tk()
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
translate_button = tk.Button(root, text="Speak", command=my_translator,font=("Arial", 10),  bg='#000', fg='#ff0',padx = 20, pady = 5)
translate_button.place(x=330,y=70)
output_text = tk.Text(root, height=10, width=50)

root.mainloop()