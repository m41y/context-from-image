#searching image for persons and items nearby to each of them, 
#as input script take image generate list of founded persons 
#and items near them as output
#for object detection pyyolo https://github.com/digitalbrain79/pyyolo, 
#wraper of yolo (darknet) is used and needed installed in system
#can be used as module or standalone script
#to use this module as standalone script run in terminal:
#python cfi.py -y <path to pyyolo> -im <path to image>
import numpy as np 
import cv2
import sys
import pyyolo

#initialize pyyolo, takes path to pyyolo folder as argument
def init_yolo(yoloPath):
    darknet_path = yoloPath+"/darknet"
    datacfg = 'cfg/coco.data'
    cfgfile = 'cfg/tiny-yolo.cfg'
    weightfile = '../tiny-yolo.weights'
    pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

#show image and wait for "q" button to exit
def showImage(img, name='frame'):
    while(True):
        cv2.imshow(name,img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  

#draw detected object on image, optionaly with names of classes and probabilities,
#takes yolos outputs and orginal image as inputs, return image with bounding boxes
def drawDetections(outputs, img, names = False, colour = 255):
    for i in range(len(outputs)):
        x,y,x1,y1 = getCoords(outputs[i])
        if names:
            className = outputs[i]["class"]
            prob = outputs[i]["prob"]
            prob = round(prob, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = str(className)+" "+"prob: "+str(prob)+"%"
            cv2.putText(img,name,(x,y), font, 0.7,(0,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x1,y1),(0,255, colour),2)
    return img

#return coordinates of detected objects as tuples of (x,y) up-rigth roi corner and (x1,y1) as bottom-left roi corner
def getCoords(outputs):
    x = outputs["left"]
    x1 = outputs["right"]
    y1 = outputs["bottom"]
    y = outputs["top"]
    return (x,y,x1,y1)

#scaling roi by the scale factor, return scaled roi, 
#(x,y) are up-right corner coordinates and (x1,y1) are bottom-left coordinates of roi bounding box
def scaleRoi(img, x,y,x1,y1, scale=1.8):
    h,w, c = img.shape
    w1 = x1-x
    h1 = y1-y
    w2 = w1*scale
    h2 = h1*scale
    dw = round((w2-w1)/2)
    dh = round((h2-h1)/2)
    x1n = int(round(x-dw))
    x2n = int(round(x1+dw))
    y1n = int(round(y-dh))
    y2n = int(round(y1+dh))
    if x1n < 0:
        x1n=0
    else:
        pass
    if x2n > w:
        x2n = w
    else:
        pass
    if y1n < 0:
        y1n = 0
    else: 
        pass
    if y2n > h:
        y2n = h
    else:
        pass
    return (x1n,y1n,x2n,y2n)

#using yolo to detect objects, returns list of dictionaries containing objects cordinates, class and probabilities
def detect(img, thresh=0.3):
    hier_thresh = 0.3
    img = img.transpose(2,0,1)
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    data = img.ravel()/255.0
    data = np.ascontiguousarray(data, dtype=np.float32)
    outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
    return outputs

#take roi and orginal image and return roi as separate image
def getObject(img, x,y,x1,y1):
    return img[y:y1, x:x1]

#resize image and keep aspect ratio
def shrink(img, max_height=1000, max_width=1000):
    height, width = img.shape[:2]
    if max_height < height or max_width < width:
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-y", "--pyyolo", help="path to pyyolo folder")
    requiredNamed.add_argument("-im", "--image", help="path to image file")
    args = vars(parser.parse_args())
#initialize pyyolo, load and shrink image
    init_yolo(args["pyyolo"])
    img = cv2.imread(args["image"])
    img = shrink(img)
    #detect items in image
    outputs = detect(img)
    #filtr outputs for persons
    persons = []
    for i in range(len(outputs)):
        if outputs[i]["class"] == "person":
            persons.append(outputs[i])  
    #for each person founded check for items near them        
    objects = []
    print " "
    print "iterating over persons..."
    print " "
    for i in range(len(persons)):
        x,y,x1,y1 = getCoords(persons[i])
        x,y,x1,y1 = scaleRoi(img, x,y,x1,y1)
        person = getObject(img, x,y,x1,y1)
        output = detect(person)
        detected = (person, output)
        objects.append(detected)
    #generate output
    print " "
    print "found "+str(len(objects))+" persons in image"
    print " "
    for i in range(len(objects)):
        print "near person "+str(i)+" found: "
        for n in range(len(objects[i][1])):
            print objects[i][1][n]["class"]