import cv2
import sys


#include tracking algorithms 

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()


#initializing webcam
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("could not open video")
    sys.exit()


ok,frame=video.read()
if not ok:
    print('cannot read video file')



#define bounding box (play around with values)
bbox=(287,23,86,320)


bbox=cv2.selectROI(frame,False)

ok = tracker.init

while True:
    #read a new frame
    ok,frame=video.read()
    if not ok:
        break

    #start timer??
    timer = cv2.getTickCount()

    #update tracker
    ok,bbox=tracker.update(frame)

    #FPS
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer); 

#drawing bounding box


# Release the webcam and close the window
video.release()
cv2.destroyAllWindows()