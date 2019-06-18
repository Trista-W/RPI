import cv2
import sys
import numpy as np



# Read video
video = cv2.VideoCapture('tracking1-100_Trim.mp4')
# Exit if video not opened.
if not video.isOpened():
    print ("Cannot open video")
    sys.exit()


# # ROI
#
# # Read first frame.
# ret, frame = video.read()
# if not ret:
#     print ("Cannot read video file")
#     sys.exit()
# tracker = cv2.TrackerKCF_create()
# boundingbox = cv2.selectROI(frame, False)
# # Initialize tracker with first frame and bounding box
# ret = tracker.init(frame, boundingbox)
#
# while True:
#     # Read a new frame
#     ret, frame = video.read()
#     if ret == False:
#         print("The end")
#         break
#     # Start timer
#     timer = cv2.getTickCount()
#     # Update tracker
#     ret, boundingbox = tracker.update(frame)
#     # Calculate Frames per second (FPS)
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
#     # Draw bounding box
#     if ret == True:
#         # Tracking success
#         p1 = (int(boundingbox[0]), int(boundingbox[1]))
#         p2 = (int(boundingbox[0] + boundingbox[2]), int(boundingbox[1] + boundingbox[3]))
#         mask = cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#         pos = (int(boundingbox[0]+boundingbox[2]/2),int(boundingbox[1]+boundingbox[3]/2))
#     else:
#         # Tracking failure
#         cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
#
#     # Display FPS on frame
#     cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
#     cv2.putText(mask, str(pos), (100,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
#     # Display result
#     cv2.imshow("Tracking", frame)
#     # Exit if ESC pressed
#     k = cv2.waitKey(1) & 0xff
#     if k == 27 : break



# # Read until video is completed
# while(video.isOpened()):
#  # Capture frame-by-frame
#  ret, frame = video.read()
#  if ret == True:
#
#    # Display the resulting frame
#    cv2.imshow('Frame',frame)
#
#    # Press Q on keyboard to  exit
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#      break
#
#  # Break the loop
#  else:
#    break
#
# # When everything done, release the video capture object
# video.release()
#
# # Closes all the frames
# cv2.destroyAllWindows()


# firstframe=None
# while True:
#
#     ret,frame = video.read()
#     if not ret:
#         break
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     gray=cv2.GaussianBlur(gray,(21,21),0)
#     if firstframe is None:
#         firstframe=gray
#         continue
#
#     frameDelta = cv2.absdiff(firstframe,gray)
#     thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
#     thresh = cv2.dilate(thresh, None, iterations=2)
#     # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
#     x,y,w,h=cv2.boundingRect(thresh)
#     frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
#     cv2.imshow("frame", frame)
#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("frame2", frameDelta)
#     key = cv2.waitKey(1)&0xFF
#
#     if key == ord("q"):
#         break
# video.release()
# cv2.destroyAllWindows()


# color range
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])

while (video != 0):
    # convert to hsv
    ret, frame = video.read()
    if ret == False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # show the originial video
    cv2.namedWindow('frame', cv2.WINDOW_FULLSCREEN)
    # cv2.imshow('frame', frame)

    # extract blue parts
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    x, y, w, h = cv2.boundingRect(mask)
    mask = cv2.rectangle(mask, (x, y), (x + w, y + h), (78, 25, 221), 2)
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(mask, "(" + str(x) + "," + str(y) + ")", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (78, 25, 221), 2)
    cv2.putText(frame, "(" + str(x) + "," + str(y) + ")", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.namedWindow('mask', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Mask', mask)
    cv2.imshow("frame", frame)

    # display the finial video
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.namedWindow('res', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('res', res)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()