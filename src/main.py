#This program has two portions. The first un-commented-out section applies Haar and MTCNN to a video and the second (commented) section applies it to webcam

#This program applies MTCNN and Haar face classifier to video inputs and detects faces.
#It was used to comparing success metrics of each classifier based on precision, recall, and frame-rate drop.



import numpy as np
import cv2
# from keras.models import load_model
import mtcnn
import time
from PIL import Image

# model = load_model('facenet_keras.h5')
#
# print(model.inputs)
# print(model.outputs)
# print(mtcnn.__version__)

#
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = mtcnn.MTCNN()

ksize = (101, 101)

# cap2 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('2021-04-25 16-59-19.mp4')
cap2.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
duration2 = cap2.get(cv2.CAP_PROP_POS_MSEC)
cap2.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

frame_number1 = 0
frame_number2 = 0
face_number1 = 0
face_number2 = 0

start2 = time.time()
while True:
    ret2, frame2 = cap2.read() #reads image data on a loop
    if ret2 == True:
        faces = detector.detect_faces(frame2) #Applies MTCNN to data and obtains faceboxes
        # detectFaceMTCNN = find_face_MTCNN(frame2, faces)
        for result in faces:
            x, y, w, h = result['box']
            x2, y2 = x +2, y+h
            # roi = frame2[y:y+h, x:x+w]
            face2 = frame2[y:y2, x:x2]
            cv2.rectangle(frame2,
                          (x, y), (x+w, y+h),
                          (0, 155, 255),
                          5)
            blur2 = cv2.GaussianBlur(face2, ksize, 0) #appleis blur to face boxes
            frame2[y:y2, x:x2] = blur2
            face_number2 +=1

        cv2.imshow('MTCNN', frame2)
        frame_number2 += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  #closes program when q is pressed or once video reaches end
            timestamp2 = cap2.get(cv2.CAP_PROP_POS_MSEC)
            break
    else:
        timestamp2 = duration2*100
        break

end2 = time.time()
cap2.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture('2021-04-25 16-59-19.mp4')
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
duration1 = cap.get(cv2.CAP_PROP_POS_MSEC)
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

start1 = time.time()
while True:
    ret, frame = cap.read()
    if ret == True:
        bboxes = classifier.detectMultiScale(frame) #applies HAAR to input data and returns boxes containing faces
        for box in bboxes:
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # print(x, y, x2, y2)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 0), 3) #draws rectanles around faces
            face1 = frame[y:y2, x:x2] #obtains coordinates of face
            blur = cv2.GaussianBlur(face1, (81, 81), 7) #applies blur

            frame[y:y2, x:x2] = blur #replaces subset of original video with blurred region
            face_number1 += 1
        frame_number1 += 1

        cv2.imshow('Haar', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #closes program when q is pressed or once video reaches end
            timestamp1 = cap.get(cv2.CAP_PROP_POS_MSEC)
            break
    else:
        timestamp1 = duration1*100
        break

end1 = time.time()

# print(timestamp1)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS of camera: ", fps, "n")


#Print Stats to Console
time2 = end2 - start2
fps2 = frame_number2/time2
print("Time alloted for MTCNN:", time2)
print("Timestamp of video upon exit:", timestamp2)
print("Slowdown by MTCNN:", (time2/timestamp2*100-100),"%")
print("Frames captured by MTCNN:", frame_number2)
print("Faces captured by MTCNN:", face_number2)
print("Percentage of frames containing faces via MTCNN:", float(face_number2/frame_number2)*100)
print("FPS of MTCNN: ", fps2, "n")
time1 = end1 - start1
fps1 = frame_number1/time1
print("Time alloted for Haars:", time1)
print("Timestamp of video upon exit:", timestamp1)
print("Slowdown by Haars:", (time1/timestamp1)*100-100,"%")
print("Frames captured by Haars:", frame_number1)
print("Faces captured by Haars:", face_number1)
print("Percentage of frames containing faces via Haars:", float(face_number1/frame_number1)*100)
print("FPS of Haars: ", fps1)
cap.release()
cv2.destroyAllWindows()



import numpy as np
import cv2
# from keras.models import load_model
import mtcnn
import time
from PIL import Image

# model = load_model('facenet_keras.h5')
#
# print(model.inputs)
# print(model.outputs)
# print(mtcnn.__version__)


# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# detector = mtcnn.MTCNN()
#
# # ksize = (101, 101)
#
# cap2 = cv2.VideoCapture(0)
#
# frame_number1 = 0
# frame_number2 = 0
# face_number1 = 0
# face_number2 = 0
#
# start2 = time.time()
# while True:
#     ret2, frame2 = cap2.read()
#     faces = detector.detect_faces(frame2)
#     # detectFaceMTCNN = find_face_MTCNN(frame2, faces)
#     for result in faces:
#         x, y, w, h = result['box']
#         x2, y2 = x +w, y+h
#         # roi = frame2[y:y+h, x:x+w]
#         face2 = frame2[y:y2, x:x2]
#         cv2.rectangle(frame2,
#                       (x, y), (x2, y2),
#                       (0, 0, 0),
#                       3)
#         blur2 = cv2.GaussianBlur(face2, (81, 81), 7)
#         frame2[y:y2, x:x2] = blur2
#         face_number2 +=1
#
#     cv2.imshow('MTCNN', frame2)
#     frame_number2 += 1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# end2 = time.time()
# cap2.release()
# # cv2.destroyAllWindows()
#
# cap = cv2.VideoCapture(0)
# start1 = time.time()
# while True:
#     ret, frame = cap.read()
#     bboxes = classifier.detectMultiScale(frame)
#     for box in bboxes:
#         x, y, width, height = box
#         x2, y2 = x + width, y + height
#         # print(x, y, x2, y2)
#         cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 0), 3)
#         face1 = frame[y:y2, x:x2]
#         blur = cv2.GaussianBlur(face1, (81, 81), 7)
#
#         frame[y:y2, x:x2] = blur
#         face_number1 += 1
#     frame_number1 += 1
#
#     cv2.imshow('Haar', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# end1 = time.time()
#
#
#
# fps = cap.get(cv2.CAP_PROP_FPS)
# print("FPS of camera: ", fps, "n")
#
# time2 = end2 - start2
# fps2 = frame_number2/time2
# print("Time alloted for MTCNN:", time2)
# print("Frames captured by MTCNN:", frame_number2)
# print("Faces captured by MTCNN:", face_number2)
# print("Percentage of frames containing faces via MTCNN:", float(face_number2/frame_number2))
# print("FPS of MTCNN: ", fps2, "n")
# time1 = end1 - start1
# fps1 = frame_number1/time1
# print("Time alloted for Haars:", time1)
# print("Frames captured by Haars:", frame_number1)
# print("Faces captured by Haars:", face_number1)
# print("Percentage of frames containing faces via Haars:", float(face_number1/frame_number1))
# print("FPS of Haars: ", fps1)
# cap.release()
# cv2.destroyAllWindows()



