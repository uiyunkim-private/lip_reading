import cv2
import keyboard
import numpy
import dlib
from imutils import face_utils
import sys
import os
import uuid
from lip_reading.network.tf_based.util import LoadLatestWeight
from lip_reading.network.tf_based.util import LR_preprocessor

def get_mouth(image, face_detector, face_predictor,mouth_shape):
    rects = face_detector(image, 0)
    if len(rects) == 0:
        return None
    for (i, rect) in enumerate(rects):
        shape = face_predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = cv2.boundingRect(numpy.array([shape[48:68]]))
        ratio = 70/ w
    image = cv2.resize(image,dsize=(0, 0),fx=ratio , fy = ratio)
    x = x * ratio
    y = y * ratio
    w = w * ratio
    h = h * ratio
    midy = y+h/2
    midx = x+w/2
    xshape = mouth_shape[1]/2
    yshape = mouth_shape[0]/2
    mouth_image = image[int(midy-yshape):int(midy+yshape), int(midx-xshape):int(midx+xshape)]
    return mouth_image

def RealtimePrediction(model,classes,shape=(120,120)):
    module_path = sys.path[1]

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(os.path.join(module_path,'lip_reading','storage','system',"shape_predictor_68_face_landmarks.dat"))
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 30);
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    info = ''
    while (capture.isOpened()):
        while True:
            ret, frame = capture.read()
            if ret:
                cv2.putText(frame, 'status: Break' , (0, 40), cv2.FONT_ITALIC, 1, (0, 0, 255),thickness=4)
                cv2.putText(frame, 'info: ' + info, (0, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                cv2.putText(frame, 'record: [space]', (0, height-80), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                cv2.putText(frame, 'exit: [ECS]', (0, height-40), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                cv2.imshow('Video Stream', frame)
                k = cv2.waitKey(1) & 0xFF
            if keyboard.is_pressed('space'):
                break
            if keyboard.is_pressed('esc'):
                capture.release()
                cv2.destroyAllWindows()
                exit(0)

        framecount = 0
        frames = []
        while framecount != 30:
            ret, frame = capture.read()
            if ret:
                frames.append(numpy.copy(frame))

                if framecount == 29:
                    cv2.putText(frame,'status: Processing', (0, 40),
                                cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                else:
                    cv2.putText(frame, 'status: RECORDING: ' + str(round((30 - framecount) / 30, 2)), (0, 40),
                                cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                cv2.imshow('Video Stream', frame)
                k = cv2.waitKey(1) & 0xFF
            framecount = framecount + 1

        faces = []
        for frame in frames:
            face = get_mouth(frame, face_detector, face_predictor, shape)
            if face is None:
                info = " NOT SAVED, " + " No face in " + str(framecount) + 'th'
                break
            faces.append(numpy.copy(face))
            cv2.imshow('face', cv2.resize(face,dsize=(0,0),fx=5,fy=5))
            k = cv2.waitKey(1) & 0xFF

        random_filename = str(uuid.uuid4())
        if(len(faces) == 30):
            video_writer = cv2.VideoWriter(os.path.join(module_path,'storage','temp',random_filename + ".mp4"), fourcc, 30, shape)
            info = 'saved [cut]'
            for frame in faces:
                video_writer.write(frame)
            video_writer.release()

        sample = LR_preprocessor(os.path.join(module_path,'storage','temp',random_filename + ".mp4"))
        data = numpy.array(sample)
        data = data.reshape(shap=(26,120,120,5))
        predictions = model.predict(data)
        print(predictions)
        count = count + 1
