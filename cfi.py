#script detect and recognise users (if users faces are in "./users" folder) and items near them,
#usage: python cfi.py -y <path to pyyolo> -im <path to image>
#as a result script will generate image of detected persons, images with each of them 
# and items near them  and file results.txt witch descriptions of founded items
#the files are named after recognised persons (if any)
#script use pyyolo which is a wrapper for yolo(darknet) for object detections and
#dlib's pretrained models for face shape and face recognition, check modules for 
#more info on dependencies
# -*- coding: utf-8 -*-
import argparse 	
import cv2
import userId as uid
import itemsNearUser as inu
import numpy as np

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("-y", "--pyyolo", help="path to pyyolo folder")
requiredNamed.add_argument("-im", "--image", help="path to image file")
args = vars(parser.parse_args())

persons = []
objects = []

print("...initializing models...")
inu.init_yolo(args["pyyolo"])
shape_path, face_rec_model_path, usersPath = uid.get_paths()
hog, shape, facerec = uid.get_predictors(shape_path, face_rec_model_path)
print("...models initialized successfully...")

print("...describing users faces...")
faces, names = uid.get_pattern_faces(usersPath)
described = uid.describe_images(faces, hog, shape, facerec)
nn = uid.fit_nn(described)
print("...users faces described successfully...")

print("...loading image and detecting persons...")
img = cv2.imread(args["image"])
img = inu.shrink(img)
outputs = inu.detect(img, thresh=0.3)
for i in range(len(outputs)):
	if outputs[i]["class"]=="person":
		persons.append(outputs[i])
numberOfPersons = len(persons)
print("...found "+str(numberOfPersons)+" persons in image...")
img2 = img.copy()
im = inu.drawDetections(persons, img)
cv2.imwrite("results/detectedPersons.jpg", im)

print("...iterating over persons...")
for i in range(len(persons)):
        x,y,x1,y1 = inu.getCoords(persons[i])
        x,y,x1,y1 = inu.scaleRoi(img2, x,y,x1,y1)
        person = inu.getObject(img2, x,y,x1,y1)
        des = uid.describe_image(person, hog, shape, facerec)
        if des != 0:
            if uid.is_verified(des[0], nn):
        	    name = uid.whois(names, des[0], nn)+str(i)
            else:
        	    name = "unknown"+str(i)
        else:
        	name = "unknown"+str(i)
        output = inu.detect(person, thresh=0.15)
        detected = (name, person, output)
        objects.append(detected)

print("...generating outputs...")
f= open('results/results.txt','w') 
for i in range(len(objects)):
    personn = "near person "+str(i)+" "+objects[i][0]+" found: "
    f.write(personn+"\n")
    print personn
    im2 = inu.drawDetections(objects[i][2], objects[i][1])
    cv2.imwrite("results/"+objects[i][0]+".jpg", im2)
    for n in range(len(objects[i][2])):
        obj = objects[i][2][n]["class"]
        f.write(obj+"\n")
        print obj
f.close() 