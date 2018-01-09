# context-from-image
Script for extracting context information from images, centered on users. The main goal is to detect users in image, if possible recognize them and detect objects near each one of them. Practical application of this script can be found in smart-space environments and in Internet of Thing domain. Using image data smart environments can obtain high level context information, even about not smart (not IoT) items. This can, for example, enable every day items that users have in their houses into their smart home settings, not to mention interfaces that can be build on top of that functionality. Usage of context information gathered from image can give "eyes" to smart spaces

<h2>Preparations<h2/>

context from image depend on pyyolo, which is a yolo (darknet) wrapper for python, that can be found [here](https://github.com/digitalbrain79/pyyolo). It's recomended to compile darknet to use gpu, so the script can execute in some rational time. If gpu is used you can use more "heavy" yolo weights other then tiny-yolo used by default to keep script to execute on cpu in less then minute. Those "heavier" weights can improve recognition and detection scores and can by found on official yolo [page](https://pjreddie.com/darknet/yolo/). After downloading new weights remember to change weightpath in itemsNearUser.py. 

Another pretrained models used by this script are dlib's models used to recognise faces. Models can be found [here](https://github.com/davisking/dlib-models), used modela are: dlib_face_recognition_resnet_model_v1.dat.bz2 and shape_predictor_68_face_landmarks.dat.bz2, just download them and extract into "./models" directory. 

Other dependencies such as opencv, sklearn or dlib can be installed using *requirements.txt* file, ie: 

*pip install -r requirements.txt*

<h2>Usage<h2/>

To use script type in terminal:

*python cfi.py -y <path to pyyolo> -im <path to image>* 
  
  this will execute script and generate results in "/results" directory. Script outputs are image with bounding boxes around found persons, images of each persons found on image with detected items near them, named after that person if recognition was sucess and .txt file containig list of items detected near each of detected persons.

