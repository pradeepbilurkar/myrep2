import os

shape_predictor_path    = os.path.join('data','shape_predictor_68_face_landmarks.dat')
alarm_paths             = [os.path.join('data','audio_files','short_horn.wav'),
                           os.path.join('data','audio_files','long_horn.wav'),
                           os.path.join('data','audio_files','distraction_alert.wav')]

EYE_DROWSINESS_THRESHOLD = .12# 0.25
EYE_DROWSINESS_INTERVAL = 0.9# 2.0
MOUTH_DROWSINESS_THRESHOLD = 0.25  #0.37
MOUTH_DROWSINESS_INTERVAL = 0.6  # 1.0
DISTRACTION_INTERVAL = 1.0 # 3

WideOpenEye_limit = 18
headDownCount = 1
sleep_count = 2
yawn_count = 2
wide_open_eye_count=0
eyesWideOpen=16


Looking_Left= 8
Looking_Right = -8
Looking_Down = -5
Looking_up = +40

