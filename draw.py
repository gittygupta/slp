import tensrflow as tf
import numpy
import cv2

# Poses
'''
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},        ###
//     {10, "RKnee"},       ###
//     {11, "RAnkle"},      ###
//     {12, "LHip"},        ###
//     {13, "LKnee"},       ###
//     {14, "LAnkle"},      ###
//     {15, "REye"},        ###
//     {16, "LEye"},        ###
//     {17, "REar"},        ###
//     {18, "LEar"},        ###
//     {19, "LBigToe"},     ###
//     {20, "LSmallToe"},   ###
//     {21, "LHeel"},       ###
//     {22, "RBigToe"},     ###
//     {23, "RSmallToe"},   ###
//     {24, "RHeel"},       ###
//     {25, "Background"}


# Hand

[[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
'''

HAND_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], 
            [0, 5], [5, 6], [6, 7], [7, 8], 
            [0, 9], [9, 10], [10, 11], [11, 12], 
            [0, 13], [13, 14], [14, 15], [15, 16], 
            [0, 17], [17, 18], [18, 19], [19, 20]]

# only the ones that are needed
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8]]

def draw(pred, size, coor='3D'):
    pred = pred[:, :-1]
    if coor == '3D'
        pred = tf.reshape(pred, (pred.shape[1], -1, 3))
    else:
        pred = tf.reshape(pred, (pred.shape[1], -1, 2))
    pose = tf.cast(pred[:, :9, :], dtype=tf.int32)
    left_hand = tf.cast(pred[:, 9:30, :], dtype=tf.int32)
    right_hand = tf.cast(pred[:, 30:, :], dtype=tf.int32)

    if coor == '3D':
        pose = pose[:, :, :-1]
        left_hand = left_hand[:, :, :-1]
        right_hand = right_hand[:, :, :-1]

    for i in range(pred.shape[0]):
        white = tf.zeros((size[0], size[1], 3), tf.int32)
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            cv2.line(white, pose[i][partFrom], pose[i][partTo], (255, 74, 0), 3)
            cv2.ellipse(white, pose[i][partFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(white, pose[i][partTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

        for pair in HAND_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]

            # left
            cv2.line(white, left_hand[i][partFrom], left_hand[i][partTo], (255, 74, 0), 3)
            cv2.ellipse(white, left_hand[i][partFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(white, left_hand[i][partTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

            # right
            cv2.line(white, right_hand[i][partFrom], right_hand[i][partTo], (255, 74, 0), 3)
            cv2.ellipse(white, right_hand[i][partFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(white, right_hand[i][partTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

        cv2.imwrite(str(i) + '.jpg', white)
