import tensorflow as tf
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

FINGER_COLORS = [(0, 0, 255),   # thumb
                (0, 255, 0),    # index
                (255, 255, 0),  # middle
                (0, 255, 255),  # ring 
                (255, 0, 255)]  # pinkie

# only the ones that are needed
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8]]

def draw(pred, size, coor='3D'):
    VEC = tf.convert_to_tensor([size[0] / 2., size[1] / 2.])
    ADD = tf.convert_to_tensor([size[0] / 2., size[1] / 2.])

    pred = pred[:, :-1]
    if coor == '3D':
        pred = tf.reshape(pred, (pred.shape[0], -1, 3))
    else:
        pred = tf.reshape(pred, (pred.shape[0], -1, 2))
    
    pose = pred[:, :9, :]
    left_hand = pred[:, 9:30, :]
    right_hand = pred[:, 30:, :]

    if coor == '3D':
        pose = pose[:, :, :-1]
        left_hand = left_hand[:, :, :-1]
        right_hand = right_hand[:, :, :-1]

    for i in range(pred.shape[0]):
        frame = np.zeros((size[0], size[1], 3), np.uint8)
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]

            pose_partFrom = tuple(pose[i][partFrom] * VEC + ADD)
            pose_partTo = tuple(pose[i][partTo] * VEC + ADD)

            if pose_partFrom == (0., 0.) or pose_partTo == (0., 0.):
                continue

            cv2.line(frame, pose_partFrom, pose_partTo, (255, 74, 0), 1)
            #cv2.ellipse(frame, pose_partFrom, (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            #cv2.ellipse(frame, pose_partTo, (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

        # left
        for j, pair in enumerate(HAND_PAIRS):
            color = FINGER_COLORS[j // 4]

            partFrom = pair[0]
            partTo = pair[1]

            left_hand_partFrom = tuple(left_hand[i][partFrom] * VEC + ADD)
            left_hand_partTo = tuple(left_hand[i][partTo] * VEC + ADD)

            if left_hand_partFrom == (0., 0.) or left_hand_partTo == (0., 0.):
                continue

            cv2.line(frame, left_hand_partFrom, left_hand_partTo, color, 1)
            #cv2.ellipse(frame, left_hand_partFrom, (2, 2), 0, 0, 360, (255, 255, 0), cv2.FILLED)
            #cv2.ellipse(frame, left_hand_partTo, (2, 2), 0, 0, 360, (255, 255, 0), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

        # right
        for j, pair in enumerate(HAND_PAIRS):
            color = FINGER_COLORS[j // 4]

            partFrom = pair[0]
            partTo = pair[1]

            right_hand_partFrom = tuple(right_hand[i][partFrom] * VEC + ADD)
            right_hand_partTo = tuple(right_hand[i][partTo] * VEC + ADD)

            if right_hand_partFrom == (0., 0.) or right_hand_partTo == (0., 0.):
                continue

            cv2.line(frame, right_hand_partFrom,right_hand_partTo, color, 1)
            #cv2.ellipse(frame, right_hand_partFrom, (2, 2), 0, 0, 360, (255, 255, 0), cv2.FILLED)
            #cv2.ellipse(frame, right_hand_partTo, (2, 2), 0, 0, 360, (255, 255, 0), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

        cv2.imwrite('output/' + str(i) + '.jpg', frame)


def main():
    sign_path = 'train'
    net_seq_len = 512
    pred = get_processed_data_2d(sign_path, [['02September_2010_Thursday_tagesschau-8371']], 0, net_seq_len)
    pred = pred[0, :50]
    draw(pred, size=(300, 300), coor='2D')

if __name__ == '__main__':
    from data_utils import *
    main()
    #cv2.imwrite('hello.jpg', np.zeros((120, 120), dtype=np.uint32) + 255)
