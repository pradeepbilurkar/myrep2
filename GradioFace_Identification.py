
def detect_face():
    import face_recognition as fr
    import os
    import cv2
    from queue import Queue
    import numpy as np
    import pickle

    path = "C:/Users/admin/Python Projects/ImageStiching/Face_Identification_Images/"
    known_names = []
    known_name_encodings = []
    images = os.listdir(path)
    # video_capture = cv2.VideoCapture(0)
    frame_queue = Queue()
    processed_frame_counter = 0
    print_interval = 10
    display_interval = 5
    # for _ in images:
    #     image = fr.load_image_file(os.path.join(path, _))
    #     encoding = fr.face_encodings(image)[0]
    #     known_name_encodings.append(encoding)
    #     known_names.append(os.path.splitext(os.path.basename(_))[0].capitalize())
    #     with open('C:/Users/admin/Python Projects/ImageStiching/known_name_encodings', 'wb') as pickle_file:
    #         pickle.dump(known_name_encodings, pickle_file)
    #     with open('C:/Users/admin/Python Projects/ImageStiching/known_names', 'wb') as pickle_file:
    #         pickle.dump(known_names, pickle_file)

    with open('C:/Users/admin/Python Projects/ImageStiching/known_name_encodings', 'rb') as pickle_file:
         known_name_encodings=pickle.load(pickle_file)
    with open('C:/Users/admin/Python Projects/ImageStiching/known_names', 'rb') as pickle_file:
        known_names= pickle.load(pickle_file)
    video_capture = cv2.VideoCapture(0)
    while True:
            ret, frame = video_capture.read()
            face_locations = fr.face_locations(frame)
            face_encodings = fr.face_encodings(frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_distances = fr.face_distance(known_name_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                if confidence > 0.45:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                #cv2.putText(frame, f"{name} - {confidence:.2f}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, f"{name}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Driver Identification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                Driver_id = 0
                cv2.destroyAllWindows()
                break
    cv2.destroyAllWindows()
    video_capture.release()

#detect_face()
