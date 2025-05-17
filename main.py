import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0; change if needed

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU.")

from helping_functions import extract_data, seq2seq, tokanize_texts
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import concurrent.futures

def main():
    # 1. Extract data
    data = extract_data()
    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    encoder_tokenizer = tokanize_texts(np.concatenate((dev_data[:,1], train_data[:,1]), axis=0))
    decoder_tokenizer = tokanize_texts(np.concatenate((dev_data[:,0], train_data[:,0]), axis=0), start_end_tokens=True)

    train_x = encoder_tokenizer.texts_to_sequences(train_data[:,1])
    train_y = decoder_tokenizer.texts_to_sequences(train_data[:,0])
    dev_x = encoder_tokenizer.texts_to_sequences(dev_data[:,1])
    dev_y = decoder_tokenizer.texts_to_sequences(dev_data[:,0])
    test_x = encoder_tokenizer.texts_to_sequences(test_data[:,1])
    test_y = decoder_tokenizer.texts_to_sequences(test_data[:,0])

    max_encoder_seq_length = max([len(seq) for seq in train_x + dev_x + test_x])
    max_decoder_seq_length = max([len(seq) for seq in train_y + dev_y + test_y])

    train_x = pad_sequences(train_x, maxlen=max_encoder_seq_length, padding='post')
    dev_x = pad_sequences(dev_x, maxlen=max_encoder_seq_length, padding='post')
    test_x = pad_sequences(test_x, maxlen=max_encoder_seq_length, padding='post')

    train_y = pad_sequences(train_y, maxlen=max_decoder_seq_length, padding='post')
    dev_y = pad_sequences(dev_y, maxlen=max_decoder_seq_length, padding='post')
    test_y = pad_sequences(test_y, maxlen=max_decoder_seq_length, padding='post')
    train_x = train_x[:1000]
    train_y = train_y[:1000]
    dev_x = dev_x[:1000]
    dev_y = dev_y[:1000]


    input_vocab_size = len(encoder_tokenizer.word_index) + 1
    output_vocab_size = len(decoder_tokenizer.word_index) + 1
    embedding_dim = 128
    hidden_units = 256
    encoder_layers = 1
    decoder_layers = 1
    dropout_rate = 0.2
    recurrent_dropout_rate = 0.2
    encoder_type = 'LSTM'
    decoder_type = 'LSTM'
    beam_width = 3

    # Prepare decoder target data as one-hot
    train_y_cat = np.eye(output_vocab_size)[train_y]
    dev_y_cat = np.eye(output_vocab_size)[dev_y]
    # print(f"train_y_cat shape: {train_y_cat.shape}")
    # print(train_y_cat)
    # Build and train model
    model = seq2seq(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=recurrent_dropout_rate,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        beam_width=beam_width
    )
    model.build_training_model()
    model.compile()
    model.fit([train_x, train_y[:, :-1]], train_y_cat[:, 1:, :], batch_size=64, epochs=1, validation_split=0.1)

    # Build inference models
    model.build_inference_model()

    start_token_idx = decoder_tokenizer.word_index.get('<start>', 1)
    end_token_idx = decoder_tokenizer.word_index.get('<end>', 0)

    predictions = model.evaluate(
        input_seqs=dev_x,
        start_token=tokenizer_out.word_index['<start>'],
        end_token=tokenizer_out.word_index['<end>'],
        max_dec_len=Y_padded.shape[1],
        batch_size=128
    )
    print(decoder_tokenizer.sequences_to_texts(predictions))
    print(f"\nFinal results:")
    print(f"Dev accuracy: {dev_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()