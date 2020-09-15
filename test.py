import tensorflow as tf
import numpy as np

from model import *
from attention import *
from data_utils import *
from bert_utils import *

# Global variables
num_decoder_blocks = 6
num_heads = 8
d_model = 256
d_ffn = 256
d_out = 154

# models
bert = Bert(max_sequence_length=80)
decoder = Decoder(num_decoder_blocks, num_heads, d_model, d_ffn, d_out)

# inference-loop
# only for a single input
def inference(model_path, sentence, net_seq_len):
    tar = tf.zeros((1, net_seq_len, d_out))

    # ckpt
    checkpoint_path = model_path
    ckpt = tf.train.Checkpoint(decoder=decoder,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Model Loaded!!')
    else:
        print('Initialised checkpoint')
    
    words, _, seq_lengths = bert(sentence[0])

    pad_mask = padding_mask(words.shape[1], seq_lengths)
    la_mask = look_ahead_mask(net_seq_len - 1)

    for i in range(net_seq_len - 1):
        pred = decoder(tar[:, :-1, :], words, la_mask, pad_mask)
        print("Counter : ", pred[0][i][-1])
        tar[0][i+1] = pred[0][i]

    return tar

# simple test
def test(model_path, sentences, path, video, net_sequence_length):
    # ckpt
    checkpoint_path = model_path
    ckpt = tf.train.Checkpoint(decoder=decoder,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Model Loaded!!')
    else:
        print('Initialised checkpoint')

    tar_inp = get_processed_data(path, video, 0, net_sequence_length)[:, :-1, :]
    words, _, seq_lengths = bert(sentences[0])

    pad_mask = padding_mask(words.shape[1], seq_lengths)
    la_mask = look_ahead_mask(tar_inp.shape[1])

    pred = decoder(tar_inp, words, la_mask, pad_mask)

    return pred

if __name__ == '__main__':
    model_path = 'models/path/to/model'
    sentence = ['german sentence']
    vid_path = 'path/to/videos'
    video = 'name of video in dataset'
    net_sequence_length = 512
    test(model_path, sentence, vid_path, video, net_sequence_length)
