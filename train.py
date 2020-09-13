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

# optim and ckpt
EPOCHS = 1000

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
# Loss
mse = tf.keras.losses.MeanSquaredError()

# train step
def train_step(bert_out, tar, seq_lengths):
    bert_out = tf.convert_to_tensor(bert_out)
    tar_inp = tar[:, :-1, :]
    tar_real = tar[:, 1:, :]

    pad_mask = padding_mask(bert_out.shape[1], seq_lengths)
    la_mask = look_ahead_mask(tar_inp.shape[1])

    with tf.GradientTape() as tape:
        pred = decoder(tar_inp, bert_out, la_mask, pad_mask)
        loss = mse(tar_real, pred)

    gradients = tape.gradient(loss, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

    print("Loss : ", loss)

# train loop
def train(sen_path, sign_path, batch_size, net_sequence_length):
    # ckpt
    checkpoint_path = './training_checkpoints'
    ckpt = tf.train.Checkpoint(decoder=decoder,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    else:
        print('Initialised checkpoint')

    # train
    f = open(sen_path, encoding='utf-8')
    sentences = f.readlines()
    sentences, dataset = sentences[1:], sentences[1:]
    for i in range(len(sentences)):
        sentences[i] = sentences[i][:-1].split('|')[-1]
        dataset[i] = dataset[i].split('|')[0]

    dataset = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
    sentences = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]

    for epoch in range(EPOCHS):
        print("Epoch : ", epoch + 1)
        for iteration in range(len(dataset)):
            tar = get_processed_data(sign_path, dataset, iteration, net_sequence_length)
            words, _, seq_lengths = bert(sentences[iteration])

            train_step(words, tar, seq_lengths)

        #if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                    ckpt_save_path))


def test_vid(model_path, sentences, path, video, net_sequence_length):
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
    sen_path = '/content/drive/My Drive/Internships/ISI/PHOENIX-2014-T.train.corpus.csv'
    sign_path = '/content/drive/My Drive/Internships/ISI/train'
    batch_size = 16
    net_sequence_length = 512
    train(sen_path, sign_path, batch_size, net_sequence_length)