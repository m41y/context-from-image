#recognizing user by comparing his face to faces in "./users" directory
#can be used either as a standalone script or as a python module
#script needs images of reference faces in one folder, called "users" and
#pretrained predictors in folder "models", that can be found on: https://github.com/davisking/dlib-models
#used models are dlib_face_recognition_resnet_model_v1.dat.bz2 and shape_predictor_68_face_landmarks.dat.bz2
#just download and extract in "./models" directory
#to use this module as standalone script run in terminal:
#python userId.py -im <path to image with face to verification>
#as a result script will output "user verified" and name of the file containing referance face
# if -im face match face in ./users directory or "user not recognized" otherwise
import cv2 
import numpy as np 
import dlib
from sklearn.neighbors import NearestNeighbors
import os

#get path to predictors and faces patterns
def get_paths():
    root = os.getcwd()
    models = root+"/models"
    shape_path = models+"/shape_predictor_68_face_landmarks.dat/data"
    face_rec_model_path = models+"/dlib_face_recognition_resnet_model_v1.dat/data"
    usersPath = root+"/users"
    return (shape_path, face_rec_model_path, usersPath)

#initialize and return predictors
def get_predictors(shape_path, face_rec_model_path):
    detector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(shape_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    return (detector, shapePredictor, facerec)

#loop through face images in "./user" folder and load them to list 
def get_pattern_faces(usersPath):
    faces = []
    names = []
    users = os.listdir(usersPath)
    for i in range(len(users)):
        if users[i].endswith(".jpg"):
            img = cv2.imread(usersPath+"/"+users[i])
            #img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            faces.append(img)
            name = users[i]
            name = name.split(".",1)[0]
            names.append(name)
    return faces, names

#find faces in image using HOG, input: image, output: detected faces
def find_faces(image, detector):
    dets = detector(image)
    if dets:
        return dets

#find face landmarks in detected faces, input: faces detected by HOG (by find_faces()) and corresponding full image,
#output: list of facial landmarks
def find_shape(det, image, shapePredictor):
    shape = shapePredictor(image, det)
    return shape

#describe faces using 3-fold deep learning descriptor, input: facial landmark and corresponding image, 
#output: 128 dimensional face description
def get_descriptor(shape, image, facerec):
    descriptor = facerec.compute_face_descriptor(image, shape)
    return descriptor

#describe faces in the image, input: image, output: 128 dimensional face description list for image
def describe_image(image, detector, shapePredictor, facerec):
    described = []
    dets = find_faces(image, detector)
    if dets is None:
        return 0
    else:
        for k, d in enumerate(dets):
            shape = find_shape(d, image, shapePredictor)
            descriptor = get_descriptor(shape, image, facerec)
            described.append(descriptor)
        return described

#describe LIST of images
def describe_images(images, detector, shapePredictor, facerec):
    described = []
    for i in range(len(images)):
        des = describe_image(images[i], detector, shapePredictor, facerec)
        described= described+des
    return described

#fit nearest neighbors classifier
def fit_nn(described):
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(described)
    return nn 

#verify user by distance in nearest neighbor, return True if user pass verification
def is_verified(desc, nn):
    pred = nn.kneighbors([desc])
    if pred[0][0][0] < 0.61:
        return True
    else:
        return False

#check if recognized face is similar to reference users faces  
def whois(names, desc, nn):
    pred = nn.kneighbors([desc])
    if pred[0][0][0] < 0.61:
        name = pred[1][0][0]
        name = names[name]
        return name
    else:
        return "unknown"

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-im", "--image", help = "path to face to verification", type=str, required=True)
    args = vars(parser.parse_args())
    
    shape_path, face_rec_model_path, usersPath = get_paths()
    hog, shape, facerec = get_predictors(shape_path, face_rec_model_path)
    faces, names = get_pattern_faces(usersPath)
    described = describe_images(faces, hog, shape, facerec)
    nn = fit_nn(described)

    img = cv2.imread(args["image"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    des = describe_image(img, hog, shape, facerec)
    print " "
    for i in range(len(des)):
        if is_verified(des[i], nn):
            print "user verified, its: "+whois(names,des[i], nn)
        else:
            print "user not recognized"

