import jetson.inference
import jetson.utils
import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    imgCuda = jetson.utils.cudaFromNumpy(img)

    detections = net.Detect(img)
    for d in detections:
        print(d)
        x1 = int(d.Left)
        y1 = int(d.Top)
        x2 = int(d.Right)
        y2 = int(d.Bottom)
        className = net.GetClassDesc(d.ClassID)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

    #img = jetson.utils.cudaToNumpy(imgCuda)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    

