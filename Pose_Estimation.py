import numpy as np
import cv2
import csv


chess_w = 8
chess_h = 6

# # Loading camera matrix as mtx, distortion coefficient as dist
# with open('camera1', 'r') as camera1:
#     reader1 = csv.reader(camera1)
#     # mtx1 = [row[0] for row in reader1]
#     # print(mtx1)
#     for index1, rows1 in enumerate(reader1):
#         if index1 == 1:
#             mtx1 = rows1[0]
#         if index1 == 2:
#             dist1 = rows1[1]
#     # print(mtx1)
#     # print(dist1)
#
# with open('camera2', 'r') as camera2:
#     reader2 = csv.reader(camera2)
#     for index2, rows2 in enumerate(reader2):
#         if index2 == 1:
#             mtx2 = rows2[0]
#         if index2 == 2:
#             dist2 = rows2[1]

camera1 = np.load('camera1.npz')
mtx1 = camera1['arr_0']
dist1 = camera1['arr_1']
camera2 = np.load('camera2.npz')
mtx2 = camera1['arr_0']
dist2 = camera1['arr_1']
# print(mtx1)
# print(dist1)

# Draw a 3D axis
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp1 = np.zeros((chess_h*chess_w, 3), np.float32)
objp2 = np.zeros((chess_h*chess_w, 3), np.float32)
objp1[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)
objp2[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)

axis1 = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
axis2 = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# Load video
images1 = cv2.VideoCapture('calibration1.mp4')

while True:
    ret1, frame1 = images1.read()
    if ret1:

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        find1, corners1_1 = cv2.findChessboardCorners(gray1, (chess_w, chess_h), None)

        if find1:
            corners1_2 = cv2.cornerSubPix(gray1, corners1_1, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors
            ret1, rvecs1, tvecs1 = cv2.solvePnP(objp1, corners1_2, mtx1, dist1)
            # cv2.solvePnPRansac

            # Project 3D points to image plane
            imgpts1, jac1 = cv2.projectPoints(axis1, rvecs1, tvecs1, mtx1, dist1)

            img1 = draw(frame1, corners1_2, imgpts1)
            cv2.imshow('img1', img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

images2 = cv2.VideoCapture('calibration2.mp4')
while True:
    ret2, frame2 = images2.read()
    if ret2:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        find2, corners2_1 = cv2.findChessboardCorners(gray2, (chess_w, chess_h), None)

        if find2:
            corners2_2 = cv2.cornerSubPix(gray2, corners2_1, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors
            ret2, rvecs2, tvecs2 = cv2.solvePnP(objp2, corners2_2, mtx2, dist2)

            # Project 3D points to image plane
            imgpts2, jac2 = cv2.projectPoints(axis2, rvecs2, tvecs2, mtx2, dist2)

            img2 = draw(frame2, corners2_2, imgpts2)
            cv2.imshow('img2', img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
