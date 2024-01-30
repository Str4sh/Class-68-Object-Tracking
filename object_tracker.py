
import numpy as np
import cv2
import math

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


video = cv2.VideoCapture("static/bb2.mp4")
state= "play"

tracker=cv2.legacy.TrackerCSRT_create()
detected=False
x_coords=[]
y_coords=[]

goal_x=220
goal_y=70
def drawbox(image, bbox):
    x=int(bbox[1][0]) 
    y=int(bbox[1][1])
    w=int(bbox[1][2])
    h=int(bbox[1][3])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    cv2.putText(image,"Tracked",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

def goalTracker(image, bbox):
    x=int(bbox[1][0]) 
    y=int(bbox[1][1])
    w=int(bbox[1][2])
    h=int(bbox[1][3])

    c1=x+int(w/2)
    c2=y+int(h/2)
    x_coords.append(c1)
    y_coords.append(c2)

    cv2.circle(image,(c1,c2),2,(0,255,0),3)

    cv2.circle(image,(int(goal_x),int(goal_y)),2,(255,0,0),3)

    dist=math.sqrt(((c1-goal_x)**2) + (c2-goal_y)**2)
    print("D:", dist)

    if(dist<=75):
         cv2.putText(image,"Goal Reached",(200,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    for i in range(len(x_coords)-1):
        cv2.circle(image,(x_coords[i],y_coords[i]),2,(0,0,255),3)

while True:
    if state == "play":
        check, image = video.read()

        if (check):
            image = cv2.resize(image, (700, 500), fx=1, fy=1)

            dimensions = image.shape[:2]
            H, W = dimensions

            if detected == False:

                blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
                yoloNetwork.setInput(blob)

                layerName = yoloNetwork.getUnconnectedOutLayersNames()
                layerOutputs = yoloNetwork.forward(layerName)

                boxes = []
                confidences = []
                classIds = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classId = np.argmax(scores)
                        confidence = scores[classId]

                        if confidence > confidenceThreshold:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY,  width, height) = box.astype('int')
                            x = int(centerX - (width/2))
                            y = int(centerY - (height/2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIds.append(classId)

                indexes = cv2.dnn.NMSBoxes(
                    boxes, confidences, confidenceThreshold, NMSThreshold)
                # print(indexes)  
                font = cv2.FONT_HERSHEY_SIMPLEX
                for i in range(len(boxes)):
                    # print(i)
                    if i in indexes:
                        # Write condition to detect the sports ball in the image
                        if labels[classIds[i]] == "sports ball":
                            x, y, w, h = boxes[i]

                            # Change the color of the box and label for every frame
                            if   i%2 == 0:
                                color = (0, 255, 0)
                            else:
                                color=(255,0,0)
                            # Draw bounding box and label on image
                            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                            tracker.init(image,boxes[i])
                            detected=True

                            # Draw the label above the box
                            label = labels[classIds[i]]
                            cv2.putText(image, label, (x,y-8), font, 0.7, color, 2)

            else:
                trackerInfo=tracker.update(image)
                success=trackerInfo
                bbox=trackerInfo
                if success:
                    drawbox(image, bbox)
                else:
                    cv2.putText(image,"Lost",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)


                goalTracker(image, bbox)
            cv2.imshow('Image', image)
            cv2.waitKey(1)

    key=cv2.waitKey(1)
    if key == 32:     #space
        print("stopped")
        break

    if key == 112:   #p
        state="pause"

    if key == 108:   #l
        state="play" 