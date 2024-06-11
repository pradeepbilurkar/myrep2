from parameters import *
from scipy.spatial import distance
from imutils import face_utils as face
from pygame import mixer
import imutils
import time
import dlib
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import face_recognition as fr
from keras.models import load_model

#
# global detector,predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Load the trained model
model = load_model('smoking_detection_model_1.h5')
# Some supporting functions for facial processing
frame_counter = 0


def get_max_area_rect(rects):
    if len(rects) == 0: return
    areas = []
    for rect in rects:
        areas.append(rect.area())
        #print(max(areas))
        #print(rects[areas.index(max(areas))])
    return rects[areas.index(max(areas))]

def get_eye_aspect_ratio(eye):
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    return (vertical_1+vertical_2)/(horizontal*2), vertical_1,vertical_2 #aspect ratio of eye

def get_mouth_aspect_ratio(mouth):
    horizontal=distance.euclidean(mouth[0],mouth[4])
    vertical=0
    for coord in range(1,4):
        vertical+=distance.euclidean(mouth[coord],mouth[8-coord])
    return vertical/(horizontal*3) #mouth aspect ratio

def identify_primary_face(fase):
    result_t = []
    # if len(fase) > 1:
    for k in range(0, len(fase)):
        result_t += [fase[k][2]]
    sorted_index = np.flip(np.argsort(result_t))
    frm = fase[sorted_index[0]]
    x = frm[0]
    y = frm[1]
    w = frm[2]
    h = frm[3]
    # for (x, y, w, h) in frm: ##fase[sorted_index[0]]:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return x,y,w,h


# Facial processing

def facial_processing():
    mixer.init()
    distracton_initlized = False
    eye_initialized      = False
    mouth_initialized    = False
    frame_counter = 0

    detector    = dlib.get_frontal_face_detector()
    predictor   = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    ls, le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs, re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    fps_timer = time.time()
    count_sl = 0
    count_yn = 0

    wide_open_eye = 0

    fps_counter = 0

    #mp_face_mesh = mp.solutions.face_mesh
    #face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    video_capture = cv2.VideoCapture(0)
    while True:
        _ , frame = video_capture.read()

        frame = imutils.resize(frame, width=900, height=500)
        fps_counter += 1
        #frame = cv2.flip(frame, 1)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #results = face_mesh.process(frame)
        #img_h, img_w, img_c = frame.shape
        # face_3d = []
        # face_2d = []
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        #################################################################################################################
        if time.time()-fps_timer >= 1.0:
            fps_to_display=fps_counter
            fps_timer = time.time()
            #fps_counter = 0
        #cv2.putText(frame, "FPS :"+str(fps_to_display), (750, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        #frame = imutils.resize(frame, width=900, height=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        rect=get_max_area_rect(rects)

        if rect!=None:
            distracton_initlized = False
            shape1 = predictor(gray, rect)
            shape = face.shape_to_np(shape1)

            leftEye = shape[ls:le]
            rightEye = shape[rs:re]
            leftEAR, LV1,LV2 = get_eye_aspect_ratio(leftEye)
            rightEAR, RV1, RV2 = get_eye_aspect_ratio(rightEye)
            #print(LV1)
            ewopen = (LV1+ LV2+RV1+RV2)/4
            if ewopen > WideOpenEye_limit:
                wide_open_eye = wide_open_eye+1
                print('EyeWideOpen')
            if ewopen > eyesWideOpen:
                eys_status = 'Eyes Wide Open'
                print('EyeWideOpen')
            else:
                eys_status = 'Eyes Normal'

            eye_aspect_ratio = (leftEAR + rightEAR) / 2.0


            #print(eye_aspect_ratio)
            if eye_aspect_ratio <= EYE_DROWSINESS_THRESHOLD:
                # print("EAR :", eye_aspect_ratio)
                # print("COst ;",EYE_DROWSINESS_THRESHOLD)
                if not eye_initialized:
                    eye_start_time = time.time()
                    eye_initialized = True
                    #print('Eye Time Start'+str(fps_counter))
                    #eye_frame_no = 0

                if time.time()-eye_start_time > EYE_DROWSINESS_INTERVAL:
                    #print("Eye_DEAWSINESS_Interval :", time.time()-eye_start_time)
                    #cv2.putText(frame, "Sleepy", (400, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    #print('Eye Time End' + str(fps_counter))
                    #(2.0)
                    # ### mouth inititialized code added by me
                    #print(time.time()-eye_start_time )
                    alarm_type = 0
                    count_sl = count_sl + 1
                    #print('Sleep'+str(count_sl))
                    eye_initialized = False
                    eye_frame_status = "Closed"
                    t = time.localtime()
                    current_time = time.strftime("%H.%M.%S", t)
                    print(current_time)
                    print("eye_status:",eye_frame_status+": "+np.str(count_sl))
                    # fase = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                    # for (x, y, w, h) in fase:
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #     cv2.putText(frame, "SLEEPY", (400,100),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                    if count_sl > 100:
                        count_sl = 0

                    # if  not distracton_initlized and not mouth_initialized and not mixer.music.get_busy():
                    #     mixer.music.load(alarm_paths[alarm_type])
                    #     mixer.music.play()
                else:
                    eye_initialized = False  #should be False
                    eye_frame_status = "Open"
                    # if not distracton_initlized and not mouth_initialized and mixer.music.get_busy():
                    #     mixer.music.stop()
            inner_lips = shape[60:68]
            mar = get_mouth_aspect_ratio(inner_lips)
            #print("mar:=", mar)
            if mar > MOUTH_DROWSINESS_THRESHOLD:

                if not mouth_initialized:
                    mouth_start_time = time.time()
                    mouth_initialized = True

                if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:  # 1
                    alarm_type=0
                    cv2.putText(frame, "Yawning", (int(frame.shape[0] / 2), 40),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), )
                    print("Yawning")
                    # fase = face_classifier.detectMultiScale(
                    #     frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                    # for (x, y, w, h) in fase:
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0,255), 2)
                    #     print("Yawning")
                    #     cv2.putText(frame, "Yawning", (int(frame.shape[0]/2), 40),
                    #                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                        # cv2.putText(frame, "YAWNING", (400,100),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    mouth_initialized = False
                    #eye_initialized =False  #### COde added by me
                    mouth_frame_status = 'Yawning'
                    count_yn = count_yn+1
                    #print('Yawning')

                    # if not mixer.music.get_busy():
                    #     mixer.music.load(alarm_paths[alarm_type])
                    #     mixer.music.play()
            else:
                mouth_initialized=False
                mouth_frame_status ='Normal'
                # if not distracton_initlized and not eye_initialized and mixer.music.get_busy():
                #     mixer.music.stop()

            if wide_open_eye> wide_open_eye_count and count_sl > sleep_count and count_yn > yawn_count:
                fase = face_classifier.detectMultiScale(
                    frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                for (x, y, w, h) in fase:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "'wide_open_eye_TIRED'", (5, int(y) + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                print('wide_open_eye_TIRED')
                wide_open_eye_TIRED= True
                wide_open_eye = 0
                count_sl = 0
                count_yn = 0
        else:
            alarm_type = 1
            if not distracton_initlized:
                distracton_start_time = time.time()
                distracton_initlized = True
            if time.time()- distracton_start_time> DISTRACTION_INTERVAL:  # 3
                cv2.putText(frame, "EYES NOT ON ROAD", (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                distracton_initlized = False

            # if not eye_initialized and not mouth_initialized and not  mixer.music.get_busy():
            #     mixer.music.load(alarm_paths[alarm_type])
            #     mixer.music.play()
            # cv2.putText(frame, "EYES NOT ON ROAD", (10, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        # cv2.imshow("Driver Monitoring", frame)
        # image = cv2.resize(frame, (224, 224))
        # image = image.astype("float") / 255.0
        # image = np.expand_dims(image, axis=0)
        #
        # # Classify the image
        # proba = model.predict(image, verbose=0)[0][0]
        # # proba = model.predict(image)[0][0]
        # # print(proba)
        # # label = "Smoking" if proba > 0.5 else "No Smoking"
        # if proba > 0.7:
        #     smk = smk + 1
        #     label = "Smoking"
        #     #cv2.putText(frame, label, (400, 100),
        #                 #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #
        # if smk >= 1:
        #     # print(f"Probability of Smoking: {proba}")
        #     # label = "Smoking" if proba > 0.5 else "No Smoking"
        #     label = "Smoking"
        #     smk = smk - 1
        # else:
        #     label = ""
        #
        # # cv2.putText(frame, label, (400, 300),
        # #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


        # if distracted == False:
        # # Draw the label and probability on the frame
        #     #if label_phone == 'Phone_Usage':
        #         #cv2.putText(frame, "Phone_Usage Status : " + label_phone, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        #     if  eye_frame_status == 'Closed':
        #         cv2.putText(frame, "Eye Status Closed : "+eye_frame_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        #         eye_frame_status = 'Open'
        #     if  mouth_frame_status == 'Yawning':
        #         cv2.putText(frame, "Mouth Status : "+mouth_frame_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        #         mouth_frame_status = 'Normal'
        #     if  eys_status == 'Eyes Wide Open':
        #         cv2.putText(frame, "Eyes Status:Eyes Wide Open : " + eys_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        #         eys_status='Normal'
        # else : #distracted == True :#and label == "Smoking":
        #      cv2.putText(frame, "DISTRACTED ", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        #      #cv2.putText(frame, "Smoking Status : " + label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # # else :
        # #     1
        # #     #cv2.putText(frame, "DISTRACTED ", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # cv2.imshow("Driver Monitoring", frame)
        # key = cv2.waitKey(1000)&0xFF
        # if key == ord("q"):
        #     break

    cv2.destroyAllWindows()
    video_capture.release()


if __name__=='__main__':
	facial_processing()


