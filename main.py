import cv2
import numpy as np
from PIL import Image
import os
import random
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="",
    password="",
    database="",
    auth_plugin='mysql_native_password'
)
cursor = mydb.cursor()
# create table users
query = "CREATE TABLE if not exists USERS (`name` VARCHAR(255) NOT NULL, `identification` VARCHAR(255) NOT NULL)"
cursor.execute(query)

# Generation of dataset


def generate_dataset():
    face_classifier = cv2.CascadeClassifier(
        "./ai project/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor = 1.3
        # Minimum neighbour = 5
        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    cap = cv2.VideoCapture(0)
    img_id = 0
    while True:
        ret, frame = cap.read()
        # print("Camera will start taking pictures now\n")
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "imagesForVis/"+str(img_id) + ".jpg"
            # file_name_path = "/Users/aditi/Desktop/ai project/data/manasa." +str(img_id)+".jpg"
            print(file_name_path)
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # (50,50) is the position of text
            # 1 is the font size
            # (0,255,0) is the color of text
            cv2.imshow("Cropped Face", face)
        if cv2.waitKey(1) == 13 or int(img_id) == 2000:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")
# generate_dataset()

# Creation of label
# path of a picture in the dataset /Users/aditi/Desktop/ai project/dataset/aditi.1.jpg


def my_label(image_name):
    name = image_name.split('.')[-3]
    # if you have two person in your dataset
    if name == "Naman":
        return np.array([1, 0])
    elif name == "Konark":
        return np.array([0, 1])


# Creation of data
# purpose is to shuffle the data
# use of tqdm to see the progress bar

def my_data():
    data = []
    for img in os.listdir("dataset"):
        path = os.path.join("dataset", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])
    print("now shuffling the data")
    random.shuffle(data)
    return data


data = my_data()

# segregation of data into train and test

train = data[:1800]
test = data[200:]
X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
print(X_train.shape)
Y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
print(X_test.shape)
Y_test = [i[1] for i in test]


# creation of model

ops.reset_default_graph()
convonet = input_data(shape=[None, 50, 50, 1])
convonet = conv_2d(convonet, 32, 5, activation='relu')
# 32 filters of size 5x5
convonet = max_pool_2d(convonet, 5)
convonet = conv_2d(convonet, 64, 5, activation='relu')
convonet = max_pool_2d(convonet, 5)
convonet = conv_2d(convonet, 128, 5, activation='relu')
convonet = max_pool_2d(convonet, 5)
convonet = conv_2d(convonet, 64, 5, activation='relu')
convonet = max_pool_2d(convonet, 5)
convonet = conv_2d(convonet, 32, 5, activation='relu')
convonet = max_pool_2d(convonet, 5)
convonet = fully_connected(convonet, 1024, activation='relu')
convonet = dropout(convonet, 0.8)
convonet = fully_connected(convonet, 2, activation='softmax')
convonet = regression(convonet, optimizer='adam',
                      learning_rate=0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convonet, tensorboard_verbose=1)
model.fit(X_train, Y_train, n_epoch=12, validation_set=(
    X_test, Y_test), show_metric=True, run_id='FRS')


cursor = mydb.cursor()
cursor.execute("SELECT name from users where identification = 'Criminal'")
res = cursor.fetchall()[0]
for result in res:
    print("Identified Criminal is: ", result)
#visualization and prediction


def data_for_visualisation():
    Vdata = []
    for img in (os.listdir('imagesForVis')):
        path = os.path.join('imagesForVis', img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        Vdata.append([np.array(img_data), img_num])
        random.shuffle(Vdata)
    return Vdata


Vdata = data_for_visualisation()
fig = plt.figure(figsize=(20, 20))
for num, data in enumerate(Vdata[0:]):
    img_data = data[0]
    y = fig.add_subplot(5, 5, num+1)
    image = img_data
    data = img_data.reshape(50, 50, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = f"Criminal - {result}"
    else:
        str_label = 'Not Criminal'

    y.imshow(image, cmap='gray')
    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
