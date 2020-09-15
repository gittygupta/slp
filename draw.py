import tensorflow as tf
import numpy as np
import cv2
import os

from mpl_toolkits import mplot3d as plt3d
import matplotlib.pyplot as plt

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

FINGER_COLORS_3D = ['green', 'red', 'cyan', 'magenta', 'yellow']

# only the ones that are needed
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8]]

def draw_3d(pose, left_hand, right_hand, i):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')

    #print(pose.shape)
    for (x, y, z) in pose:
        if x < -0.8:
            continue
        ax.scatter(x, y, z, color='black', marker='s')
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        ### special cases
        # hands shouldd not go ridiculously across the screen
        # when model predicts values near 0
        if abs(pose[partFrom][0] - pose[partTo][0]) >= 1.:      # with normalized coords
            continue

        #if pose[partFrom] == (0., 0.) or pose_partTo == (0., 0.):
        #    continue

        xs = pose[partFrom][0], pose[partTo][0]
        ys = pose[partFrom][1], pose[partTo][1]
        zs = pose[partFrom][2], pose[partTo][2]

        line = plt3d.art3d.Line3D(xs, ys, zs)

        ax.add_line(line)
    
    # left
    for j, pair in enumerate(HAND_PAIRS):
        color = FINGER_COLORS_3D[j // 4]

        partFrom = pair[0]
        partTo = pair[1]

        ### special cases
        # when model predicts values near 0
        if left_hand[partFrom][0] < -0.8 or left_hand[partTo][0] < -0.8:
            continue

        xs = left_hand[partFrom][0], left_hand[partTo][0]
        ys = left_hand[partFrom][1], left_hand[partTo][1]
        zs = left_hand[partFrom][2], left_hand[partTo][2]

        line = plt3d.art3d.Line3D(xs, ys, zs, color=color)

        ax.add_line(line)

    # right
    for j, pair in enumerate(HAND_PAIRS):
        color = FINGER_COLORS_3D[j // 4]

        partFrom = pair[0]
        partTo = pair[1]

        ### special cases
        # when model predicts values near 0
        if right_hand[partFrom][0] < -0.8 or right_hand[partTo][0] < -0.8:
            continue

        xs = right_hand[partFrom][0], right_hand[partTo][0]
        ys = right_hand[partFrom][1], right_hand[partTo][1]
        zs = right_hand[partFrom][2], right_hand[partTo][2]

        line = plt3d.art3d.Line3D(xs, ys, zs, color=color)

        ax.add_line(line)
    
    ax.view_init(-90, -90)      # to position the axis properly
    plt.axis('off')
    plt.savefig('output/' + str(i) + '.jpg')    # use numbers only to save files
    plt.close()


def draw(pred):
    pred = pred[:, :-1]
    pred = tf.reshape(pred, (pred.shape[0], -1, 3))
    
    pose = pred[:, :9, :]
    left_hand = pred[:, 9:30, :]
    right_hand = pred[:, 30:, :]

    #draw_3d(pose, left_hand, right_hand)

    for i in range(pred.shape[0]):
        draw_3d(pose[i], left_hand[i], right_hand[i], i)
    '''
    for i in range(pred.shape[0]):
        frame = np.zeros((size[0], size[1], 3), np.uint8)
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]

            pose_partFrom = tuple(pose[i][partFrom] * VEC + ADD)
            pose_partTo = tuple(pose[i][partTo] * VEC + ADD)

            ### special cases
            # hands shouldd not go ridiculously across the screen
            # when model predicts values near 0
            if abs(pose_partFrom[0] - pose_partTo[0]) >= size[0] / 2.:
                continue

            if pose_partFrom == (0., 0.) or pose_partTo == (0., 0.):
                continue

            cv2.line(frame, pose_partFrom, pose_partTo, (255, 74, 0), 2)
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

            ### special cases
            # any hand should not be present on the top left corner
            # when model predicts values near 0
            if left_hand_partFrom[0] < size[0] / 10. or left_hand_partTo[0] < size[0] / 10.:
                continue

            if left_hand_partFrom == (0., 0.) or left_hand_partTo == (0., 0.):
                continue

            cv2.line(frame, left_hand_partFrom, left_hand_partTo, color, 2)
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

            cv2.line(frame, right_hand_partFrom,right_hand_partTo, color, 2)
            #cv2.ellipse(frame, right_hand_partFrom, (2, 2), 0, 0, 360, (255, 255, 0), cv2.FILLED)
            #cv2.ellipse(frame, right_hand_partTo, (2, 2), 0, 0, 360, (255, 255, 0), cv2.FILLED)
            #cv2.putText(white, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(white, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)

        cv2.imwrite('output/' + str(i) + '.jpg', frame)
    '''

# assume images are numbered only
def out_video(inputpath, outputpath, fps):
    image_array = []
    files = os.listdir(inputpath)
    for i in range(len(files)):
        img = cv2.imread(inputpath + '/' + str(i) + '.jpg')
        size =  (img.shape[1], img.shape[0])
        img = cv2.resize(img, size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath, fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()


def main():
    sign_path = 'train'
    net_seq_len = 512
    pred = get_processed_data(sign_path, [['02September_2010_Thursday_tagesschau-8371']], 0, net_seq_len)
    pred = pred[0, :139]
    draw(pred)

    inputpath = 'output'
    outpath =  'video.mp4'
    fps = 24
    out_video(inputpath, outpath, fps)

if __name__ == '__main__':
    from data_utils import *
    main()
    #cv2.imwrite('hello.jpg', np.zeros((120, 120), dtype=np.uint32) + 255)
