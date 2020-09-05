import tensorflow as tf
import numpy as np

def look_ahead_mask(size=512):
    return (1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)) * -1e9

def dot_product_attention(q, k, v, mask):
    attn = tf.matmul(q, k, transpose_b=True)
    if mask is not None:
        attn += mask
    attn = tf.nn.softmax(attn, axis=-1)
    attn = tf.matmul(attn, v)
    return attn

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, d_model=256):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.out = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.num_heads, self.depth))
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def call(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn = dot_product_attention(q, k, v, mask)

        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, (attn.shape[0], attn.shape[1], self.d_model))
        attn = self.out(attn)

        return attn

def main():
    x = tf.random.normal((32, 512, 140))
    y = tf.random.normal((32, 64, 768))
    mask = look_ahead_mask(512)

    mha1 = MultiHeadAttention(8, 256)
    mha2 = MultiHeadAttention(8, 256)

    # self MHA
    x = mha1(x, x, x, mask)
    print(x.shape)

    # word-frame MHA
    x = mha2(x, y, y, None)
    print(x.shape)

if __name__ == '__main__':
    main()