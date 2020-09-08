import tensorflow as tf
import numpy as np
import os

def get_data(path, dataset, iteration):
    #dataset = os.listdir(path)
    #dataset = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
    
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
            #print(filename)
            f = open(data_path + '/' + filename)
            content = f.readlines()
            
            if len(content) != 67:  # printing the ones that have discrepency of 2 persons being detected
                #print(d, filename)
                continue
            
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

            
            pose = tf.expand_dims(tf.convert_to_tensor(pose), axis=0)
            left_hand = tf.expand_dims(tf.convert_to_tensor(left_hand), axis=0)
            right_hand = tf.expand_dims(tf.convert_to_tensor(right_hand), axis=0)

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
    seq_lengths = []
    for i, (poses, left_hands, right_hands) in enumerate(data):
        poses = tf.reshape(poses[:, :, :], (poses.shape[0], -1))     # removed 3rd column
        left_hands = tf.reshape(left_hands[:, :, :], (left_hands.shape[0], -1))
        right_hands = tf.reshape(right_hands[:, :, :], (right_hands.shape[0], -1))
        #poses = tf.reshape(poses, (poses.shape[0], -1))[:, :-poses.shape[1]]
        #left_hands = tf.reshape(left_hands, (left_hands.shape[0], -1))[:, :-left_hands.shape[1]]
        #right_hands = tf.reshape(right_hands, (right_hands.shape[0], -1))[:, :-right_hands.shape[1]]

        positions = positional_encoding(poses.shape[0])     # passing number of frames
        concat_data = tf.concat([poses, left_hands, right_hands, tf.expand_dims(positions, axis=-1)], axis=-1)
        start_token = tf.zeros((1, concat_data.shape[-1]))
        concat_data = tf.concat([start_token, concat_data], axis=0)

        seq_lengths.append(concat_data.shape[1])

        padded_data = tf.expand_dims(padding(concat_data, net_sequence_length), axis=0)

        if i == 0:
            processed_data = padded_data
        else:
            processed_data = tf.concat([processed_data, padded_data], axis=0)

    #processed_data = tf.reshape(processed_data, (len(data), net_sequence_length, -1))   
    return processed_data, seq_lengths


def get_processed_data(path, dataset, iteration, net_sequence_length):
    batched_data = get_data(path, dataset, iteration)
    preprocessed_data, seq_lengths = preprocess(batched_data, net_sequence_length)

    return preprocessed_data, seq_lengths

