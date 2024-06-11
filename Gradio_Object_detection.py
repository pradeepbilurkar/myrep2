def gradio_object():
    import cv2
    import cvlib as cv
    from cvlib.object_detection import draw_bbox
    #
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        bbox, label, conf = cv.detect_common_objects(frame, confidence= 0.2, model="yolov3-tiny")
        output_image = draw_bbox(frame, bbox, label, conf)
        cv2.imshow("Frame", output_image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
             break
