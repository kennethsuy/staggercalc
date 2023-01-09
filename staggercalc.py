# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import json
import argparse
import sys
import numpy as np
import os.path
#import matplotlib.pyplot as plt
#import time
print(cv.__version__)


parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
#parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--screenshots', action="store_true")

args = parser.parse_args()

def drawLabel(image, x, y, name):
    cv.putText(image, name, (x - 20, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):

    #contour for uhc box
    #purple only
    lower = np.array([127, 125, 195])
    upper = np.array([132, 130, 200])
#    lower = np.array([0, 125, 150])
#    upper = np.array([132, 255, 200])
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    output = cv.bitwise_and(frame,frame, mask= mask)
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    (t, binary) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
#    cv.imshow(winName, binary)
#    return

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("no countours")
        return
    cnt = contours[0]

    max_area = cv.contourArea(cnt)

    for cont in contours:
        
        if cv.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv.contourArea(cont)

    cv.drawContours(frame, [cnt], 0, (0,255,0), 1)
    
    x,y,w,h = cv.boundingRect(cnt)

    drawLabel(frame,x,y, "stagger bar")
    if(h != 5):
        print("height: " + str(h)+"; should be 5, usually.")
    print("width: " + str(w))
    print("Stagger: " +str(532-w))
#    cv.imshow(winName, binary) 
    return 532-w

def drawLabel(image, x, y, name):
    cv.putText(image, name, (x - 20, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.namedWindow(winName)
namearray=[]
if(args.screenshots):
    for namefile in [f for f in os.listdir("screenshots") if (f.endswith('.png') or f.endswith('.jpg'))]:
        if not os.path.isfile("screenshots/"+namefile):
            print("Input image file ", args.image, " doesn't exist")
        else:
            namearray.append("screenshots/"+namefile)
elif (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    namearray.append(args.image)
else:
    print("Input image file doesn't exist")
    sys.exit(1)

outputFile="default.jpg"
outputFolder="calculatedstagger/"


for n in namearray:
    print(n)
    if args.screenshots:
        filename = n[12:-4]
    else:
        filename = n[:-4]
    frame = cv.imread(n)
    # get frame from the video
    if not (args.image or args.screenshots):
        frame = cv.imread(n)

    framecopy = frame.copy()
    framewidth = int(frame.shape[1])
    frameheight = int(frame.shape[0])
    print("framesize: ")
    print(framewidth)
    print(frameheight)
    #print()

    outs = "this is a test for contour only"
    showtime = postprocess(frame, outs)
    #cv.imshow(winName, frame)

    if showtime != None:
        if(args.screenshots):
            outputFileNow=outputFolder+filename+"px_"+str(showtime)+".jpg"
            print(outputFileNow)
            cv.imwrite(outputFileNow, frame.astype(np.uint8))
        elif (args.image):
            outputFileNow=filename+"px_"+str(showtime)+".jpg"
            print(outputFileNow)
            cv.imwrite(outputFileNow, frame.astype(np.uint8))
        
