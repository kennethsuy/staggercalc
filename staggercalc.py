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
parser.add_argument('--video', help='Path to video file.')

args = parser.parse_args()


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
    print("width: " + str(w))
    print("Stagger: " +str(532-w))
#    cv.imshow(winName, binary) 
    return 532-w


# Process inputs
winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.namedWindow(winName)
outputFile = "check_results.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    hasFrame, frame = cap.read()
    outputFile = args.image[:-4]+'_check_results.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_check_results.avi'
else:
    print("Input image file doesn't exist")
    sys.exit(1)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))




while cv.waitKey(1) < 0:
    
    # get frame from the video
    if not (args.image):
        hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    framecopy = frame.copy()
    framewidth = int(frame.shape[1])
    frameheight = int(frame.shape[0])
    print("framesize: ")
    print(framewidth)
    print(frameheight)
    #print()

    # # Create a 4D blob from a frame.
    # blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # # Sets the input to the network
    # net.setInput(blob)

    # # Runs the forward pass to get output of the output layers
    # outs = net.forward(getOutputsNames(net))

    # # Remove the bounding boxes with low confidence
    outs = "this is a test for contour only"
    showtime = postprocess(frame, outs)
    cv.imshow(winName, frame)

    if showtime != None:
        print("contour found")


    cv.waitKey(5000)
    frame=framecopy
        #is check more than 1/8th the photo but less than 95% of the photo
        #if checkheight*checkwidth * 8 > frameheight * framewidth and checkheight*checkwidth * 1.05 < frameheight * framewidth:
        #    cv.waitKey(3000)
            #yesno = input("Does this look good to you? (y)? ")
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    #t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # Write the frame with the detection boxes

"""     if (args.image):
        v.imwrite(outputFile, detections.astype(np.uint8))
        #retval, framejpg = cv.imencode(".jpg",frame)
        #b64txt = base64.b64encode(framejpg)
        #b64txt = b64txt.decode("utf-8")
        #print(b64txt)
    else:
        vid_writer.write(frame.astype(np.uint8))
 """