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

        poses = tf.convert_to_tensor([])
        left_hands = tf.convert_to_tensor([])
        right_hands = tf.convert_to_tensor([])
        for i, filename in enumerate(files):
            f = open(filename)
            content = f.readlines()
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

            
            pose = tf.convert_to_tensor(pose)
            left_hand = tf.convert_to_tensor(left_hand)
            right_hand = tf.convert_to_tensor(right_hand)

            if i == 0:
                poses = pose
                left_hands = left_hand
                right_hands = right_hand

            else:
                poses = tf.concat([poses, pose], axis=0)
                left_hands = tf.concat([left_hands, left_hand], axis=0)
                right_hands = tf.concat([right_hands, right_hand], axis=0)

        poses = tf.reshape(poses, (i+1, -1, poses.shape[-1]))
        left_hands = tf.reshape(left_hands, (i+1, -1, left_hands.shape[-1]))
        right_hands = tf.reshape(right_hands, (i+1, -1, right_hands.shape[-1]))

        batched_data.append([poses, left_hands, right_hands])

    return batched_data


def padding(data, net_sequence_length):
    pad_length = net_sequence_length - data.shape[0]
    pad = tf.zeros((pad_length, data.shape[1]))
    return tf.concat([data, pad], axis=0)


def preprocess(data, net_sequence_length):   # removing 3rd column and concatenating all
    processed_data = tf.convert_to_tensor([])
    for i, (poses, left_hands, right_hands) in enumerate(data):
        poses = tf.reshape(poses, (poses.shape[0], -1))[:, :-poses.shape[1]]
        left_hands = tf.reshape(left_hands, (left_hands.shape[0], -1))[:, :-left_hands.shape[1]]
        right_hands = tf.reshape(right_hands, (right_hands.shape[0], -1))[:, :-right_hands.shape[1]]

        concat_data = tf.concat([poses, left_hands, right_hands], axis=-1)
        padded_data = padding(concat_data, net_sequence_length)

        if i == 0:
            processed_data = padded_data
        else:
            processed_data = tf.concat([processed_data, padded_data], axis=0)

    processed_data = tf.reshape(processed_data, (len(data), net_sequence_length, -1))
        

    return processed_data


def get_processed_data(path, batch_size, iteration, net_sequence_length):
    dataset = os.listdir(path)
    dataset = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]

    batched_data = get_data(path, dataset, iteration)
    preprocessed_data = preprocess(batched_data, net_sequence_length)

    return preprocessed_data