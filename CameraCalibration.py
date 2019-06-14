import cv2
import numpy as np
import csv


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# chessboard verticies = 86, 9x7 squares
chess_w = 8
chess_h = 6


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_h * chess_w, 3), np.float32)
objp[:, :2] = np.mgrid[0: chess_w, 0: chess_h].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

frame_num = 0
images = cv2.VideoCapture('calibration2.mp4')
save = 0

while True:
    ret, frame = images.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        # height, width = gray.shape[::-1]
        find, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
        if find:
            frame_num = frame_num + 1
            if frame_num % 10 == 0:   # calibration1 - 30, calibration2 - 10

                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cali = cv2.drawChessboardCorners(frame, (7, 6), corners2, ret)
                cv2.imshow('saved', cali)
                save = save + 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# cv2.destroyAllWindows()
print(frame_num)
print(save)



ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# output camera matrix as .csv file
with open('camera2', 'w') as csvfile:
    fieldnames = ['Camera Matrix', 'Distortion Coefficient', 'Rotation Vectors', 'Translation Vectors']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Camera Matrix': mtx})
    writer.writerow({'Distortion Coefficient': dist})
    writer.writerow({'Rotation Vectors': rvecs})
    writer.writerow({'Translation Vectors': tvecs})

# Undistortion
images2 = cv2. VideoCapture('tracking2.mp4')
while True:
    ret2, frame2 = images2.read()
    if ret2:
        height, width = frame2.shape[:2]
        newcameramtx, roi = cv2. getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
        dst = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('undistorted', dst)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

images.release()
images2.release()
cv2.destroyAllWindows()


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2. projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2[i])
    mean_error += error

print('total error: {}'.format(mean_error/len(objpoints)))
