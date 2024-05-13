import cv2
import numpy as np
import face_recognition
import os

path = r"C:\Users\CompuMart\Desktop\computer vision\pics"          # the folder that have our images
images = []              # for the images that we will import
imagesName= []           # contain the name of thr images that we appear under the name


# save files name
list = os.listdir(path)

for i in list:
    
    image = cv2.imread(f'{path}/{i}')
    images.append(image)
    # imagesName.append(i)

    # will split the extention and the root
    imagesName.append(os.path.splitext(i)[0])

# print(imagesName)

def encoding(images):
    encoded = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = face_recognition.face_encodings(img)[0]
        encoded.append(img)
    return encoded


encoded_List = encoding(images)

camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(frame)
    encode = face_recognition.face_encodings(frame, faces)

    for facencode, faceloc in zip(encode, faces):
        matching = face_recognition.compare_faces(encoded_List, facencode)
        distance = face_recognition.face_distance(encoded_List, facencode)
        # print(distance)
        matches_image = np.argmin(distance)

        if matching[matches_image]:
            name = imagesName[matches_image]
            print(name)
            y1, x2, y2, x1 = faceloc               #b  g   r 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2+35, y2), (0, 0, 255),cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,0.6,(255, 255, 255),2)

    
    cv2.imshow("Video", img)

    if cv2.waitKey(1) == ord('x'):
        break