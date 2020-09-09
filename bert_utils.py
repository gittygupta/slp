import tensorflow as tf
import numpy as np
import bert

#max_sequence_length = 64

class Bert(object):
    def __init__(self, max_sequence_length):     # doesn't quite matter
        super(Bert, self).__init__()
        self.max_seq_length = max_sequence_length
        self.model_dir = 'models/multi_cased_L-12_H-768_A-12'
        self.model_ckpt = 'models/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
        self.vocab_file = 'models/multi_cased_L-12_H-768_A-12/vocab.txt'

        self.model = self.create_bert_model()
        self.tokenizer = self.create_bert_tokenizer()

    def create_bert_model(self):
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        l_input_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32')
        output = l_bert(l_input_ids)
        model = tf.keras.Model(inputs=l_input_ids, outputs=output)
        model.build(input_shape=(None, self.max_seq_length))

        bert.load_stock_weights(l_bert, self.model_ckpt)

        return model

    def create_bert_tokenizer(self):
        model_name = 'multi_cased_L-12_H-768_A-12'
        do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
        bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, self.model_ckpt)
        tokenizer = bert.bert_tokenization.FullTokenizer(self.vocab_file, do_lower_case)

        return tokenizer

    def tokenize(self, sequence):
        sequence = self.tokenizer.tokenize(sequence)
        return ["[CLS]"] + sequence + ["[SEP]"]

    def get_sequence_ids(self, tokenized_sequence):
        sequence_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sequence)
        #print(len(sequence_ids))   Use as padding_mask
        seq_length = len(sequence_ids)
        sequence_ids = sequence_ids + [0] * (self.max_seq_length - len(sequence_ids))    # padding
        return sequence_ids, seq_length

    def preprocess(self, sequence):
        return self.get_sequence_ids(self.tokenize(sequence))

    def preprocess_batch(self, sequence_list):
        if type(sequence_list) != list:
            sequence_list = [sequence_list]

        preprocessed_sequence_ids = []
        seq_lengths = []
        for sequence in sequence_list:
            sequence_ids, seq_length = self.preprocess(sequence)
            preprocessed_sequence_ids.append(tf.convert_to_tensor(sequence_ids))
            seq_lengths.append(seq_length)

        #print(preprocessed_sequence_ids)
        preprocessed_sequence_ids = tf.convert_to_tensor(preprocessed_sequence_ids)
        #print(preprocessed_sequence_ids)
        #print(preprocessed_sequence_ids.shape)

        return preprocessed_sequence_ids, seq_lengths

    def __call__(self, sequence_list):
        #if type(sequence_list) != list or sequence_list:
        #    sequence_list = [sequence_list]
        preprocessed_sequence_ids, seq_lengths = self.preprocess_batch(sequence_list)
        #print(preprocessed_sequence_ids)
        output = self.model.predict(preprocessed_sequence_ids)      # (?, 64, 768)  for all sentences in batch
        word_embeddings = output                                    # (?, 64, 768)
        
        sentence_embeddings = []
        for sentence in output:
            sentence_vector = sentence[0]                           # feature vector for [CLS] is the sentence vector for each sentence
            sentence_embeddings.append(sentence_vector)

        sentence_embeddings = np.array(sentence_embeddings)         # (?, 768)

        return word_embeddings, sentence_embeddings, seq_lengths

'''
def main():
    bert_model = Bert(64)
    word_embeddings, sentence_embeddings = bert_model.predict(['sonst wechselhaft mit schauern und gewittern die uns auch am wochenende begleiten', 
                                                            'und nun die wettervorhersage für morgen donnerstag den zwölften august'])
    print(word_embeddings.shape, sentence_embeddings.shape)

if __name__ == "__main__":
    main()
'''