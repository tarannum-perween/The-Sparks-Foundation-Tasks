#Object Detection using SSD-MobileNetv3
#Implementation using Python and OpenCV.

import cv2

thres = 0.5  #threshold to detect object
cap = cv2.VideoCapture(0)    #Capture video by the default camera 

#captured window setup 
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)


classlabel = []

#Training data set
classfile = 'coco.names.txt'
with open(classfile, "rt") as f:
    classlabel = f.read().rstrip('\n').split('\n')

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox= model.detect(img, confThreshold = thres)
    print(classIds)
    if(len(classIds) != 0):
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if(classId<=80):
                cv2.rectangle(img, box,color=(0,255,0), thickness = 3)
                cv2.putText(img, classlabel[classId-1].upper(), (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.putText(img, str(round(confidence*100,2)), (box[0]+200,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)


    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
