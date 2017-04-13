import numpy as np
import random
import cv2
import glob

#Initialize fisher face/eigen classifier
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
#emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
face = cv2.face.createFisherFaceRecognizer()
#face = cv2.face.createEigenFaceRecognizer()

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            #open image
            image = cv2.imread(item)
            #convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #append image array to training data list
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        #repeat above process for prediction set
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    face.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = face.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))

#Now run it
metascore = []
for i in range(0,10):
    correct = run_recognizer()
    print "got", correct, "percent correct!"
    metascore.append(correct)

print "\n\nend score:", np.mean(metascore), "percent correct!"