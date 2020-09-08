import tensorflow as tf
from attention import *

class FFN(tf.keras.layers.Layer):
    def __init__(self, d_ffn, d_model):
        super(FFN, self).__init__()
        self.d1 = tf.keras.layers.Dense(d_ffn)
        self.d2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ffn):
        super(DecoderBlock, self).__init__()
        
        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)

        self.ffn = FFN(d_ffn, d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()

    def call(self, x, bert_out, look_ahead_mask, padding_mask):
        attn = self.mha1(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn)

        attn = self.mha2(x, bert_out, bert_out, padding_mask)
        x = self.layernorm2(x + attn)

        ffn_out = self.ffn(x)
        x = self.layernorm3(x + ffn_out)

        return x


class Decoder(tf.keras.Model):
    def __init__(self, num_decoder_blocks, num_heads, d_model, d_ffn, d_out):
        super(Decoder, self).__init__()

        self.num_decoder_blocks = num_decoder_blocks

        # 1st layer
        self.init_layer = tf.keras.layers.Dense(d_model)

        self.decoder_blocks = []
        for _ in range(num_decoder_blocks):
            self.decoder_blocks.append(DecoderBlock(num_heads, d_model, d_ffn))

        # final layer
        self.final_layer = tf.keras.layers.Dense(d_out)#, activation='relu')

    def call(self, x, bert_out, look_ahead_mask, padding_mask):
        x = self.init_layer(x)

        for i in range(self.num_decoder_blocks):
            x = self.decoder_blocks[i](x, bert_out, look_ahead_mask, padding_mask)
        
        x = self.final_layer(x)

        return x



def main():
    import random

    batch_size = 32

    x = tf.random.normal((batch_size, 512, 256))
    bert_out = tf.random.normal((batch_size, 64, 768))

    lookaheadmask = look_ahead_mask(512)
    paddingmask = padding_mask(64, random.sample(range(0, 64), batch_size))

    decoder = Decoder(num_decoder_blocks=6, num_heads=8, d_model=256, d_ffn=256, d_out=144)
    
    out = decoder(x, bert_out, lookaheadmask, paddingmask)

    print(decoder.summary())
    print(out)

if __name__ == '__main__':
    main()