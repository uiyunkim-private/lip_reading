import cv2
import keyboard
import numpy
import dlib
from imutils import face_utils
import sys
import os
import uuid

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

def Dataset_recorder(class_name = 'nolabel',type='train',shape=(120,120),save_original=True):
    module_path = sys.path[1]
    if class_name == 'nolabel':
        cut_dataset_path = os.path.join(module_path ,'dataset','cut',class_name)
    else:
        cut_dataset_path = os.path.join(module_path, 'dataset', 'cut',type, class_name)
    os.makedirs(cut_dataset_path, exist_ok=True)
    if save_original:
        if class_name == 'nolabel':
            original_dataset_path = os.path.join(module_path,'dataset' ,'original', class_name)
        else:
            original_dataset_path = os.path.join(module_path, 'dataset', 'original', type, class_name)
        os.makedirs(original_dataset_path, exist_ok=True)
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(os.path.join(module_path,'system',"shape_predictor_68_face_landmarks.dat"))
    capture = cv2.VideoCapture(1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = 0
    info = ''
    while (capture.isOpened()):
        while True:
            ret, frame = capture.read()
            if ret:
                cv2.putText(frame, 'Status: Break'+ info, (0, 40), cv2.FONT_ITALIC, 1, (0, 0, 255),thickness=4)
                cv2.putText(frame, 'record: [space]', (0, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                cv2.putText(frame, 'exit: [ECS]', (0, 120), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
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
                    cv2.putText(frame,'Status: Processing', (0, 25),
                                cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                else:
                    cv2.putText(frame, 'Status: RECORDING: ' + str(round((30 - framecount) / 30, 2)), (0, 25),
                                cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=4)
                cv2.imshow('Video Stream', frame)
                k = cv2.waitKey(1) & 0xFF
            framecount = framecount + 1

        random_filename = str(uuid.uuid4())

        faces = []
        for frame in frames:
            face = get_mouth(frame, face_detector, face_predictor, shape)
            faces.append(numpy.copy(face))
            cv2.imshow('face', face)
            k = cv2.waitKey(1) & 0xFF
            if face is None:
                info = "NOT SAVED, " + " No face in " + str(framecount) + 'th'
                break


        if(len(faces) == 30):
            video_writer = cv2.VideoWriter(os.path.join(cut_dataset_path,random_filename + ".mp4"), fourcc, 30, shape)
            info = 'saved [cut]'
            for frame in faces:
                video_writer.write(frame)
            video_writer.release()
            count = count + 1
        if save_original:
            if (len(frames) == 30):
                video_writer = cv2.VideoWriter(os.path.join(original_dataset_path, random_filename + ".mp4"), fourcc, 30,(int(width), int(height)))
                info = info +' [original]'
                for frame in frames:
                    video_writer.write(frame)
                video_writer.release()
                count = count + 1

