# Background Subtraction using pybgs library
# This script demonstrates the usage of the pybgs library for background subtraction in a video.
# It processes a video file, extracts the moving objects by applying the FrameDifference algorithm,
# and displays the original video, the foreground mask, and the background model in real-time.
#



# Import necessary libraries
import numpy as np
import cv2
import pybgs as bgs
#algorithm = bgs.FrameDifference()
# Initialize the background subtraction algorithm
algorithm = bgs.SuBSENSE()
video_file = "/home/gaelmarquez/usdot-gradecrossing-monotoring/bgs_playground/myData/clip_01.mov"

# Configure the blob detector once (re-creating it every frame is wasteful)
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255          # foreground mask is white blobs on black
params.filterByArea = True      # enable so minArea is actually applied
params.minArea= 100   # toy around with these values.
#params.maxArea = some large value ?? wonder if i can do this more testing will be done
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False  # corrected spelling (was filterByIntertia)

detector = cv2.SimpleBlobDetector_create(params)

# Create a video capture object to read the video file
capture = cv2.VideoCapture(video_file)

# Wait for the video file to be opened
while not capture.isOpened():
  capture = cv2.VideoCapture(video_file)
  cv2.waitKey(1000)
  print("Waiting for the video to be loaded...")


# Main loop to process the video frames
while True:
  # Read a new frame
  flag, frame = capture.read()
   #getting frame rate data
  fps=capture.get(cv2.CAP_PROP_FPS)
  print('Frames per second : ' , fps,'FPS')

    #get frame count
  frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
  print('Frame Count: ',frame_count)
  
  # If a frame was successfully read
  if flag:
    # Display the original video frame
    #cv2.imshow('Original Video', frame)
    # Apply the background subtraction algorithm
    img_output = algorithm.apply(frame)
    # Retrieve the current background model
    img_bgmodel = algorithm.getBackgroundModel()

    # Display the foreground mask and the background model
    cv2.imshow('Foreground Mask', img_output)
    #cv2.imshow('Background Model', img_bgmodel)

    # Run blob detection on the foreground mask (not the video path)
    mask = img_output
    if mask.ndim == 3:
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    keypoints = detector.detect(mask)

    # Draw detected blobs on the original frame so they're easy to see
    vid_with_keypoints = cv2.drawKeypoints(
        frame, keypoints, np.array([]), (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Blob Detection', vid_with_keypoints)

  else:
    # Wait for a bit and exit the loop if no frame is captured
    cv2.waitKey(1000)
    print("No more frames to read or error in reading the frame.")
    break
  
  # Break the loop if the user presses 'Esc'
  if cv2.waitKey(10) & 0xFF == 27:
    print("Exiting...")
    break

# Clean up: close all OpenCV windows and release the video capture object
cv2.destroyAllWindows()
capture.release()

