import os
import gc

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

from helping_functions import extract_data, tokanize_texts, add_start_end, seq2seq, BahdanauAttention, Seq2SeqAttention, export_parallel_visualization_html, plot_decoder_activations_multi_combined

import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import concurrent.futures
import wandb
import json
import random
import argparse
from keras.models import Model
import pandas as pd

def get_config_value(config, args, key, default=None):
    return getattr(config, key, getattr(args, key, default))

def main(args):     
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)

    data = extract_data(get_config_value(wandb.config, args, 'dataset'))
    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    config = wandb.config

    # Set up the tokenizer for the encoder and decoder
    encoder_tokenizer = tokanize_texts(np.concatenate((dev_data[:,1], train_data[:,1]), axis=0))
    decoder_tokenizer = tokanize_texts(np.concatenate((dev_data[:,0], train_data[:,0]), axis=0), start_end_tokens=True)

    # Convert texts to sequences
    train_y = decoder_tokenizer.texts_to_sequences(add_start_end(train_data[:,0]))
    dev_y = decoder_tokenizer.texts_to_sequences(add_start_end(dev_data[:,0]))
    test_y = decoder_tokenizer.texts_to_sequences(add_start_end(test_data[:,0]))

    train_x = encoder_tokenizer.texts_to_sequences(train_data[:,1])
    dev_x = encoder_tokenizer.texts_to_sequences(dev_data[:,1])
    test_x = encoder_tokenizer.texts_to_sequences(test_data[:,1])

    max_encoder_seq_length = max([len(seq) for seq in train_x + dev_x])
    max_decoder_seq_length = max([len(seq) for seq in train_y + dev_y])

    # Pad sequences to the same length
    train_x = pad_sequences(train_x, maxlen=max_encoder_seq_length, padding='post')
    dev_x = pad_sequences(dev_x, maxlen=max_encoder_seq_length, padding='post')
    test_x = pad_sequences(test_x, maxlen=max_encoder_seq_length, padding='post')

    train_y = pad_sequences(train_y, maxlen=max_decoder_seq_length, padding='post')
    dev_y = pad_sequences(dev_y, maxlen=max_decoder_seq_length, padding='post')
    test_y = pad_sequences(test_y, maxlen=max_decoder_seq_length, padding='post')

    input_vocab_size = len(encoder_tokenizer.word_index) + 1
    output_vocab_size = len(decoder_tokenizer.word_index) + 1
    train_y_cat = np.eye(output_vocab_size)[train_y]

    run_name = f"{get_config_value(config, args, 'cell_type')}_{get_config_value(config, args, 'attention')}_{get_config_value(config, args, 'beam_width')}_{get_config_value(config, args, 'embedding_dim')}_{get_config_value(config, args, 'hidden_units')}_learning_rate_{get_config_value(config, args, 'learning_rate')}"
    wandb.run.name=run_name
    model = None
    
    # Check if attention is enabled
    if get_config_value(config, args, 'attention'):
        model = Seq2SeqAttention(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            embedding_dim=get_config_value(config, args, 'embedding_dim'),
            hidden_units=get_config_value(config, args, 'hidden_units'),
            dropout_rate=get_config_value(config, args, 'dropout_rate'),
            recurrent_dropout_rate=get_config_value(config, args, 'recurrent_dropout_rate'),
            encoder_type=get_config_value(config, args, 'cell_type'),
            decoder_type=get_config_value(config, args, 'cell_type'),
            beam_width=get_config_value(config, args, 'beam_width')
        )
    
    else:
        model = seq2seq(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            embedding_dim=get_config_value(config, args, 'embedding_dim'),
            hidden_units=get_config_value(config, args, 'hidden_units'),
            encoder_layers=get_config_value(config, args, 'encoder_layers'),
            decoder_layers=get_config_value(config, args, 'decoder_layers'),
            dropout_rate=get_config_value(config, args, 'dropout_rate'),
            recurrent_dropout_rate=get_config_value(config, args, 'recurrent_dropout_rate'),
            encoder_type=get_config_value(config, args, 'cell_type'),
            decoder_type=get_config_value(config, args, 'cell_type'),
            beam_width=get_config_value(config, args, 'beam_width')
        )
    
    model.build_training_model()
    optimizer = keras.optimizers.Adam(
        learning_rate=get_config_value(config, args, 'learning_rate')
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x=[train_x, train_y[:, :-1]],
        y=train_y_cat[:, 1:, :],
        batch_size=400,
        epochs=get_config_value(config, args, 'epochs'),
        validation_split=0.0
    )
    model.build_inference_model()

    dev_y_eval = dev_y[:, 1:]
    test_y_eval = test_y[:, 1:]

    if get_config_value(config, args, 'attention_extractor') or get_config_value(config, args, 'connectivity'):
        model.build_attention_extractor()
        output_seq, attn_matrices = model.get_attention_for_example(
            input_seq=test_x[0],  # shape: (input_len,)
            start_token=decoder_tokenizer.word_index.get('§', 1),
            end_token=decoder_tokenizer.word_index.get('¶', 0),
            max_dec_len=dev_y_eval.shape[1]
        )

        attention_data = []
        indices = [random.randint(0, len(test_x)-1) for _ in range(9)]

        reverse_decoder_map = {v: k for k, v in decoder_tokenizer.word_index.items()}
        reverse_decoder_map[0] = ''
        reverse_encoder_map = {v: k for k, v in encoder_tokenizer.word_index.items()}
        reverse_encoder_map[0] = ''

        def tokens_to_text_list(tokens, reverse_map, remove_special=None):
            if remove_special is None:
                remove_special = []
            return [reverse_map.get(tok, '') for tok in tokens if tok > 0 and reverse_map.get(tok, '') not in remove_special]

        for idx in indices:
            input_seq = test_x[idx]
            output_seq, attn_matrices = model.get_attention_for_example(
                input_seq=input_seq,
                start_token=decoder_tokenizer.word_index.get('§', 1),
                end_token=decoder_tokenizer.word_index.get('¶', 0),
                max_dec_len=dev_y.shape[1]
            )
            input_labels = tokens_to_text_list(input_seq, reverse_encoder_map)
            output_labels = tokens_to_text_list(output_seq, reverse_decoder_map, remove_special=['§', '¶', '<e>'])
            attn_list = attn_matrices.tolist()  # convert numpy array to list for JSON
        
            attention_data.append({
                "idx": idx,
                "input_labels": input_labels,
                "output_labels": output_labels,
                "attention": attn_list
            })            

        with open(f"attention_data_{get_config_value(config, args, 'cell_type')}.jsonl", "w", encoding="utf-8") as f:
            for item in attention_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        if get_config_value(config, args, 'connectivity'):
            export_parallel_visualization_html(0, out_path="attention_comparison.html")
            wandb.log({"attention_comparison": wandb.Html("attention_comparison.html")})

    if get_config_value(config, args, 'LSTM_cell_vis') and get_config_value(config, args, 'cell_type') == 'LSTM':
        decoder_layer_name = "decoder_0"   # name of the first decoder LSTM
        decoder_lstm_layer = model.training_model.get_layer(decoder_layer_name)
        sequence_tensor = decoder_lstm_layer.output[0]  # pick the first output (the full sequence)
    
        intermediate_decoder_model = Model(
            inputs=model.training_model.inputs,   # still [encoder_input, decoder_input]
            outputs=sequence_tensor               # just the (batch, T_dec, hidden_units) tensor
        )

        num_examples = 15
        num_neurons = 5
        x_enc = np.expand_dims(test_x[0], axis=0)
        x_dec_in = np.expand_dims(test_y[0], axis=0)
        decoder_seq_outputs = intermediate_decoder_model.predict([x_enc, x_dec_in], verbose=0)  # (1, T_dec, hidden_units)
        hidden_units = decoder_seq_outputs.shape[2]
        # Pick random indices from test set
        random_indices = random.sample(range(len(test_x)), num_examples)
        # Pick random neuron indices (make sure < hidden_units)
        random_neurons = random.sample(range(hidden_units), num_neurons)

        examples_data = []
        reverse_decoder_map = {index: char for char, index in decoder_tokenizer.word_index.items()}
        reverse_decoder_map[0] = ""

        for idx in random_indices:
            x_enc = np.expand_dims(test_x[idx], axis=0)
            x_dec_in = np.expand_dims(test_y[idx], axis=0)
            decoder_seq_outputs = intermediate_decoder_model.predict([x_enc, x_dec_in], verbose=0)  # (1, T_dec, hidden_units)
            T_dec = decoder_seq_outputs.shape[1]
            # Get decoded chars (remove padding and special tokens)
            seq = test_y[idx]
            decoded_chars = [reverse_decoder_map.get(tok, '') for tok in seq if tok > 0 and reverse_decoder_map.get(tok, '') not in ['§', '¶']]
            # Extract activations for selected neurons
            activations = {}
            for nidx in random_neurons:
                activations[str(nidx)] = decoder_seq_outputs[0, :, nidx].tolist()
            examples_data.append({
                "idx": idx,
                "decoded_chars": decoded_chars,
                "neuron_indices": random_neurons,
                "activations": activations,
                "T_dec": T_dec
            })
        with open("decoder_activations_multi.json", "w", encoding="utf-8") as f:
            json.dump(examples_data, f, ensure_ascii=False, indent=2)
        
        plot_decoder_activations_multi_combined("decoder_activations_multi.json")

    # Evaluate the model on the validation and test sets
    if get_config_value(config, args, 'do_val'):  
        dev_acc, dev_prediction = model.evaluate(
            input_seqs=dev_x,
            target_seqs=dev_y_eval,
            start_token=decoder_tokenizer.word_index.get('§', 1),
            end_token=decoder_tokenizer.word_index.get('¶', 0),
            max_dec_len=dev_y_eval.shape[1],
            batch_size=1000
        )
        wandb.log({"validation_accuracy": dev_acc})

    if get_config_value(config, args, 'do_test'):
        test_acc, test_prediction = model.evaluate(
            input_seqs=test_x,
            target_seqs=test_y_eval,
            start_token=decoder_tokenizer.word_index.get('§', 1),
            end_token=decoder_tokenizer.word_index.get('¶', 0),
            max_dec_len=dev_y_eval.shape[1],
            batch_size=1000
        )
        wandb.log({"test_accuracy": test_acc})

        folder_name = "predictions_attention" if get_config_value(config, args, 'attention') else "predictions_vanilla"
        os.makedirs(folder_name, exist_ok=True)

        # Reverse mapping from token to char
        reverse_decoder_map = {v: k for k, v in decoder_tokenizer.word_index.items()}
        reverse_decoder_map[0] = ''  # padding
        reverse_encoder_map = {v: k for k, v in encoder_tokenizer.word_index.items()}
        reverse_encoder_map[0] = ''  # padding

        def tokens_to_text(tokens, reverse_map, remove_special=None):
            if remove_special is None:
                remove_special = []
            return ''.join([reverse_map.get(tok, '') for tok in tokens if tok > 0 and reverse_map.get(tok, '') not in remove_special])

        # Convert test_x, test_y, and predictions to text
        x_texts = [tokens_to_text(seq, reverse_encoder_map) for seq in test_x]
        y_true_texts = [tokens_to_text(seq, reverse_decoder_map, remove_special=['§', '¶']) for seq in test_y]
        y_pred_texts = [tokens_to_text(seq, reverse_decoder_map, remove_special=['§', '¶']) for seq in test_prediction]

        df = pd.DataFrame({'x': x_texts, 'true_y': y_true_texts, 'pred_y': y_pred_texts})
        csv_path = os.path.join(folder_name, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script that return model weights")
    parser.add_argument(
        '-we',
        '--wandb_entity',
        type=str,
        default=None,
        help='Wandb Entity used to track experiments in the Weights & Biases dashboard'
    )
    parser.add_argument(
        '-wp',
        '--wandb_project',
        type=str,
        default=None,
        help='Project name used to track experiments in Weights & Biases dashboard'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default="path/to/training/data",
        help='give the train directory location of your dataset'
    )
    parser.add_argument(
        '-ct',
        '--cell_type',
        type=str,
        default="LSTM",
        help='Type of cell to use in the model (LSTM, GRU, etc.)'
    )
    parser.add_argument(
        '-att',
        '--attention',
        type=bool,
        default=False,
        help='Use attention mechanism (True/False)'
    )
    parser.add_argument(
        '-att_ex',
        '--attention_extractor',
        type=bool,
        default=False,
        help='Use attention extractor (True/False)'
    )
    parser.add_argument(
        '-conn',
        '--connectivity',
        type=bool,
        default=False,
        help='Use connectivity (True/False)'
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=400,
        help='Batch size for training'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '-emb',
        '--embedding_dim',
        type=int,
        default=256,
        help='Dimensionality of the embedding layer'
    )
    parser.add_argument(
        '-hu',
        '--hidden_units',
        type=int,
        default=256,
        help='Number of hidden units in the LSTM/GRU layers'
    )
    parser.add_argument(
        '-dr',
        '--dropout_rate',
        type=float,
        default=0.2,
        help='Dropout rate for the LSTM/GRU layers'
    )
    parser.add_argument(
        '-rec_dr',
        '--recurrent_dropout_rate',
        type=float,
        default=0.2,
        help='Recurrent dropout rate for the LSTM/GRU layers'
    )
    parser.add_argument(
        '-enc_layers',
        '--encoder_layers',
        type=int,
        default=1,
        help='Number of layers in the encoder'
    )
    parser.add_argument(
        '-dec_layers',
        '--decoder_layers',
        type=int,
        default=1,
        help='Number of layers in the decoder'
    )
    parser.add_argument(
        '-beam',
        '--beam_width',
        type=int,
        default=1,
        help='Beam width for beam search decoding'
    )
    parser.add_argument(
        '-do_val',
        '--do_val',
        type=bool,
        default=False,
        help='Evaluate on validation set (True/False)'
    )
    parser.add_argument(
        '-do_test',
        '--do_test',
        type=bool,
        default=False,
        help='Evaluate on test set (True/False)'
    )
    args = parser.parse_args()
    main(args)