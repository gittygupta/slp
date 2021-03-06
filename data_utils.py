import tensorflow as tf
import numpy as np
import os

IMG_X = 210 / 2.
IMG_Y = 260 / 2.
IMG_Z = 0.5

VEC = tf.expand_dims(tf.convert_to_tensor([IMG_X, IMG_Y, IMG_Z]), axis=-1)
SUB = tf.expand_dims(tf.convert_to_tensor([IMG_X, IMG_Y, IMG_Z]), axis=-1)

def get_data(path, dataset, iteration):
    #dataset = os.listdir(path)
    #dataset = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]

    doubles = open('doubles2.txt', 'a')
    
    data = dataset[iteration]
    
    batched_data = []
    for d in data:
        data_path = path + '/' + d
        files = os.listdir(data_path)
        files = [filename for filename in files if filename[-3:] == 'txt']
        files.sort()
        
        poses = tf.convert_to_tensor([])
        left_hands = tf.convert_to_tensor([])
        right_hands = tf.convert_to_tensor([])

        #print(d)
        for i, filename in enumerate(files):
            f = open(data_path + '/' + filename)
            content = f.readlines()
            f.close()

            if len(content) == 0:   # special case
                print(d, filename)
                doubles.write(d + ' ' + filename + '\n')
                continue

            if len(content) != 67:  # printing the ones that have discrepency of > 1 persons being detected
                pose_count = 25
                hand_count = 21

                skips = (len(content) % 67) // 3
            
                count = 0

                pose = []
                left_hand = []
                right_hand = []

                for j in range(pose_count):
                    if j == pose_count - 1:
                        pose.append(list(map(float, content[j][3:-3].split())))
                    else:
                        pose.append(list(map(float, content[j][3:-2].split())))
                
                count = j + skips * (pose_count + 1)
                for j in range(count + 1, count + hand_count + 1):
                    if j == (count + hand_count):
                        left_hand.append(list(map(float, content[j][3:-3].split())))
                    else:
                        left_hand.append(list(map(float, content[j][3:-2].split())))
        
                count = j + skips * (hand_count + 1)
                for j in range(count + 1, count + hand_count + 1):
                    if j == (count + hand_count):
                        right_hand.append(list(map(float, content[j][3:-3].split())))
                    else:
                        right_hand.append(list(map(float, content[j][3:-2].split())))


            else:            
                pose = []
                left_hand = []
                right_hand = []
                for j, con in enumerate(content):
                    if j < 25:
                        if j == 24:
                            pose.append(list(map(float, con[3:-4].split())))
                        else:
                            pose.append(list(map(float, con[3:-2].split())))
                    elif j < 46:
                        if j == 45:
                            left_hand.append(list(map(float, con[3:-4].split())))
                        else:
                            left_hand.append(list(map(float, con[3:-2].split())))
                    else:
                        if j == len(content) - 1:
                            right_hand.append(list(map(float, con[3:-4].split())))
                        else:
                            right_hand.append(list(map(float, con[3:-2].split())))

            ### Normalize ###
            pose = tf.transpose((tf.transpose(tf.convert_to_tensor(pose)) - SUB) / VEC)
            left_hand = tf.transpose((tf.transpose(tf.convert_to_tensor(left_hand)) - SUB) / VEC)
            right_hand = tf.transpose((tf.transpose(tf.convert_to_tensor(right_hand)) - SUB) / VEC)
            ###

            pose = tf.expand_dims(pose, axis=0)
            left_hand = tf.expand_dims(left_hand, axis=0)
            right_hand = tf.expand_dims(right_hand, axis=0)

            #print("-----------POSE : -----------", pose)
            #print("------------LEFT : ----------", left_hand)
            #print("------------RIGHT: ----------",right_hand)

            if poses.shape == (0,):
                poses = pose
                left_hands = left_hand
                right_hands = right_hand

            else:
                #print(poses.shape)
                #print(pose.shape)
                poses = tf.concat([poses, pose], axis=0)
                left_hands = tf.concat([left_hands, left_hand], axis=0)
                right_hands = tf.concat([right_hands, right_hand], axis=0)

        #poses = tf.reshape(poses, (i+1, -1, poses.shape[-1]))
        #left_hands = tf.reshape(left_hands, (i+1, -1, left_hands.shape[-1]))
        #right_hands = tf.reshape(right_hands, (i+1, -1, right_hands.shape[-1]))

        batched_data.append([poses, left_hands, right_hands])

    doubles.close()

    return batched_data


def positional_encoding(seq_length):    # values between 0 and 1 for each frame
    counter = []
    for i in range(seq_length):
        counter.append(i / float(seq_length - 1))
    return tf.convert_to_tensor(counter)


def padding(data, net_sequence_length):
    pad_length = net_sequence_length - data.shape[0]
    pad = tf.zeros((pad_length, data.shape[1]))
    return tf.concat([data, pad], axis=0)

def preprocess(data, net_sequence_length):   # removing 3rd column and concatenating all
    processed_data = tf.convert_to_tensor([])
    #seq_lengths = []
    for i, (poses, left_hands, right_hands) in enumerate(data):
        # poses [9 to 24 removed]
        poses = np.delete(poses, list(range(9, 25)), axis=1)
        poses = tf.reshape(poses[:, :, :], (poses.shape[0], -1))
        
        left_hands = tf.reshape(left_hands[:, :, :], (left_hands.shape[0], -1))
        right_hands = tf.reshape(right_hands[:, :, :], (right_hands.shape[0], -1))
        #poses = tf.reshape(poses, (poses.shape[0], -1))[:, :-poses.shape[1]]
        #left_hands = tf.reshape(left_hands, (left_hands.shape[0], -1))[:, :-left_hands.shape[1]]
        #right_hands = tf.reshape(right_hands, (right_hands.shape[0], -1))[:, :-right_hands.shape[1]]

        positions = positional_encoding(poses.shape[0])     # passing number of frames
        concat_data = tf.concat([poses, left_hands, right_hands, tf.expand_dims(positions, axis=-1)], axis=-1)
        start_token = tf.zeros((1, concat_data.shape[-1]))
        concat_data = tf.concat([start_token, concat_data], axis=0)

        padded_data = tf.expand_dims(padding(concat_data, net_sequence_length), axis=0)

        if i == 0:
            processed_data = padded_data
        else:
            processed_data = tf.concat([processed_data, padded_data], axis=0)

    #processed_data = tf.reshape(processed_data, (len(data), net_sequence_length, -1))   
    return processed_data#, seq_lengths
    

def get_processed_data(path, dataset, iteration, net_sequence_length):
    batched_data = get_data(path, dataset, iteration)
    preprocessed_data = preprocess(batched_data, net_sequence_length)

    return preprocessed_data

