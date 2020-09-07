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
d_out = 202

# models
bert = Bert()
decoder = Decoder(num_decoder_blocks, num_heads, d_model, d_ffn, d_out)

EPOCHS = 1000

# Optim
learning_rate = 0.0001  # Following microsoft
optimizer = tf.keras.optimizers.Adam(learning_rate)     # rest default

# ckpt
checkpoint_path = './training_checkpoints'
ckpt = tf.train.Checkpoint(decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

# Loss
mse = tf.keras.losses.MeanSquaredError()

# train step
def train_step(bert_out, tar, seq_lengths):
    bert_out = tf.convert_to_tensor(bert_out)
    tar_inp = tar[:, :-1, :]
    tar_real = tar[:, 1:, :]

    #pad_mask = padding_mask(tar_inp.shape[1], seq_lengths)
    # padding mask will be none for now because output comes from bert
    la_mask = look_ahead_mask(tar_inp.shape[1])

    with tf.GradientTape() as tape:
        pred = decoder(tar_inp, bert_out, la_mask, None)
        loss = mse(tar_real, pred)

    gradients = tape.gradient(loss, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

    print("Loss : ", loss)

# train loop
def train(sen_path, sign_path, batch_size, net_sequence_length):
    f = open(sen_path)
    sentences = f.readlines()
    sentences, dataset = sentences[1:], sentences[1:]
    for i in range(len(sentences)):
        sentences[i] = sentences[i][:-1].split('|')[-1]
        dataset[i] = dataset[i].split('|')[0]

    dataset = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
    sentences = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]

    for epoch in range(EPOCHS):
        for iteration in range(len(dataset)):
            tar, seq_lengths = get_processed_data(sign_path, dataset, iteration, net_sequence_length)
            bert_out = bert(sentences[iteration])[0]

            train_step(bert_out, tar, seq_lengths)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

        #print("Epoch : {}, Loss : {}")


if __name__ == '__main__':
    sen_path = '/content/drive/My Drive/Internships/ISI/PHOENIX-2014-T.train.corpus.csv'
    sign_path = '/content/drive/My Drive/Internships/ISI/train'
    batch_size = 16
    net_sequence_length = 512
    train(sen_path, sign_path, batch_size, net_sequence_length)