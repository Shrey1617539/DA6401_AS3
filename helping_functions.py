import numpy
import os
from tensorflow import keras
import numpy as np
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Concatenate, Layer
)
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import random
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Nirmala UI' 

def extract_data(data_location = '/kaggle/input/as3-dataset/lexicons'):
    # Helper function to load a TSV file and return as numpy array
    def load_tsv(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(line.strip().split('\t'))
        return numpy.array(data, dtype=object)

    data = {}
    # Load train, dev, and test splits from the specified directory
    for split in ['train', 'dev', 'test']:
        file_name = f"gu.translit.sampled.{split}.tsv"
        file_path = os.path.join(data_location, file_name)
        data[split] = load_tsv(file_path)
    
    return data

def tokanize_texts(texts, char_level=True, start_end_tokens=False):
    # Optionally add start/end tokens to each text
    START_TOKEN = '§'
    END_TOKEN = '¶'
    if start_end_tokens:
        texts = [START_TOKEN + text + END_TOKEN for text in texts]
    # Create a Keras tokenizer at character level (no filtering, case-sensitive)
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=char_level, filters='', lower=False)
    tokenizer.fit_on_texts(texts)
    return tokenizer
    
def add_start_end(texts):
    # Add start and end tokens to each text in the list
    START_TOKEN = '§'
    END_TOKEN = '¶'
    return [START_TOKEN + text + END_TOKEN for text in texts]

class seq2seq:
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim,
        hidden_units,
        encoder_layers,
        decoder_layers,
        dropout_rate,
        recurrent_dropout_rate,
        encoder_type,
        decoder_type,
        beam_width
    ):
        # Store model hyperparameters for later use
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.beam_width = beam_width
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
    
    def build_training_model(self):
        # Encoder input and embedding layer
        encoder_inputs = keras.layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = keras.layers.Embedding(
            input_dim=self.input_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)

        encoder_states = []
        encoder_outputs = encoder_embedding

        # Stack encoder RNN layers (LSTM/GRU/RNN)
        for i in range(self.encoder_layers):
            return_sequences = (i < self.encoder_layers - 1)  # Only last layer returns last state
            return_state = True

            if self.encoder_type == 'LSTM':
                rnn_layer = keras.layers.LSTM(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'encoder_{i}'
                )
                encoder_outputs, state_h, state_c = rnn_layer(encoder_outputs)
                encoder_states.extend([state_h, state_c])
            elif self.encoder_type == 'GRU':
                rnn_layer = keras.layers.GRU(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'encoder_{i}'
                )
                encoder_outputs, state_h = rnn_layer(encoder_outputs)
                encoder_states.append(state_h)
            elif self.encoder_type == 'SimpleRNN':
                rnn_layer = keras.layers.SimpleRNN(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'encoder_{i}'
                )
                encoder_outputs, state_h = rnn_layer(encoder_outputs)
                encoder_states.append(state_h)
        
        # Decoder input and embedding layer
        decoder_inputs = keras.layers.Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = keras.layers.Embedding(
            input_dim=self.output_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)

        decoder_outputs = decoder_embedding
        decoder_init_states = []

        # Prepare initial decoder states from encoder final states
        idx = 0
        for i in range(self.decoder_layers):
            if i < self.encoder_layers:
                if self.decoder_type == 'LSTM':
                    h = encoder_states[idx]
                    c = encoder_states[idx + 1]
                    decoder_init_states.append([h, c])
                    idx += 2
                else:
                    h = encoder_states[idx]
                    decoder_init_states.append([h])
                    idx += 1
            else:
                # If decoder has more layers than encoder, repeat last encoder state
                if self.decoder_type == 'LSTM':
                    h = encoder_states[-2]
                    c = encoder_states[-1]
                    decoder_init_states.append([h, c])
                else:
                    h = encoder_states[-1]
                    decoder_init_states.append([h])

        # Stack decoder RNN layers (LSTM/GRU/RNN)
        for i in range(self.decoder_layers):
            return_sequences = True
            return_state = True

            if self.decoder_type == 'LSTM':
                rnn_layer = keras.layers.LSTM(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )
                decoder_outputs, _, _ = rnn_layer(decoder_outputs, initial_state=decoder_init_states[i])
            elif self.decoder_type == 'GRU':
                rnn_layer = keras.layers.GRU(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )
                decoder_outputs, _ = rnn_layer(decoder_outputs, initial_state=decoder_init_states[i])
            elif self.decoder_type == 'SimpleRNN':
                rnn_layer = keras.layers.SimpleRNN(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )
                decoder_outputs, _ = rnn_layer(decoder_outputs, initial_state=decoder_init_states[i])

        # Output layer: projects decoder outputs to vocabulary size
        decoder_dense = keras.layers.Dense(
            units=self.output_vocab_size,
            activation='softmax',
            name='decoder_dense'
        )
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the full training model (encoder + decoder)
        self.training_model = keras.models.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=decoder_outputs
        )
    
    def build_inference_model(self):
        # Build encoder inference model for prediction (single step at a time)
        encoder_inputs = keras.layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding_layer = self.training_model.get_layer('encoder_embedding')
        encoder_embedding = encoder_embedding_layer(encoder_inputs)

        encoder_outputs = encoder_embedding
        encoder_states = []
        # Run through encoder layers to get final states for inference
        for i in range(min(self.encoder_layers, self.decoder_layers)):
            encoder_rnn_layer = self.training_model.get_layer(f'encoder_{i}')
            encoder_outputs, *state = encoder_rnn_layer(encoder_outputs)
            encoder_states.extend(state)

        # Encoder model outputs all encoder states needed for decoder initialization
        self.encoder_model = keras.models.Model(
            inputs=encoder_inputs,
            outputs=encoder_states
        )

        # Build decoder inference model for step-by-step prediction
        decoder_inputs = keras.layers.Input(shape=(None,), name='decoder_inputs')
        decoder_embedding_layer = self.training_model.get_layer('decoder_embedding')
        decoder_embedding = decoder_embedding_layer(decoder_inputs)

        decoder_states_inputs = []
        # Prepare input placeholders for decoder's initial states at each layer
        for idx, state in enumerate(encoder_states):
            decoder_states_inputs.append(
                keras.layers.Input(shape=(self.hidden_units,), name=f'decoder_state_input_{idx}')
            )

        decoder_outputs = decoder_embedding
        decoder_states = []

        state_idx = 0
        # Rebuild decoder RNN stack for inference, using state inputs
        for i in range(self.decoder_layers):
            decoder_rnn_layer = self.training_model.get_layer(f'decoder_{i}')
            if i < self.encoder_layers:
                if self.decoder_type == 'LSTM':
                    init_h = decoder_states_inputs[state_idx]
                    init_c = decoder_states_inputs[state_idx + 1]
                    decoder_outputs, state_h, state_c = decoder_rnn_layer(
                        decoder_outputs, initial_state=[init_h, init_c]
                    )
                    decoder_states.extend([state_h, state_c])
                    state_idx += 2
                else:
                    init_h = decoder_states_inputs[state_idx]
                    decoder_outputs, state_h = decoder_rnn_layer(
                        decoder_outputs, initial_state=[init_h]
                    )
                    decoder_states.append(state_h)
                    state_idx += 1
            else:
                # For extra decoder layers, repeat last encoder state
                if self.decoder_type == 'LSTM':
                    init_h = decoder_states_inputs[-2]
                    init_c = decoder_states_inputs[-1]
                    decoder_outputs, state_h, state_c = decoder_rnn_layer(
                        decoder_outputs, initial_state=[init_h, init_c]
                    )
                    decoder_states.extend([state_h, state_c])
                else:
                    init_h = decoder_states_inputs[-1]
                    decoder_outputs, state_h = decoder_rnn_layer(
                        decoder_outputs, initial_state=[init_h]
                    )
                    decoder_states.append(state_h)

        decoder_dense_layer = self.training_model.get_layer('decoder_dense')
        decoder_outputs = decoder_dense_layer(decoder_outputs)

        # Final decoder model for inference: takes decoder input and previous states, outputs next token and new states
        self.decoder_model = keras.models.Model(
            inputs=[decoder_inputs] + decoder_states_inputs,
            outputs=[decoder_outputs] + decoder_states
        )

    def compile(self, optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy']):
        # Compile the training model with optimizer, loss, and metrics
        self.training_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, x, y, batch_size=64, epochs=10, validation_split=0):
        # Train the model on the provided data
        self.training_model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
        )

    def evaluate(
        self,
        input_seqs,
        target_seqs,
        start_token,
        end_token,
        max_dec_len,
        batch_size=64):
        """
        Batched beam search decoding + exact‐match accuracy.
        Uses one big GPU call per time‐step over all (batch×beam) hypotheses.
        """
        N = input_seqs.shape[0]
        n_batches = math.ceil(N / batch_size)
        total_correct = 0
        predictions = []

        for bi in range(n_batches):
            # Prepare batch for this iteration
            batch_inputs = input_seqs[bi*batch_size : (bi+1)*batch_size]
            bsz = batch_inputs.shape[0]

            # Get encoder states for the batch
            enc_states = self.encoder_model.predict(batch_inputs, verbose=0)

            B = self.beam_width
            flat_states = []
            # Tile encoder states for beam search (repeat for each beam)
            for state in enc_states:
                tiled = np.repeat(state[:, None, :], B, axis=1)
                flat_states.append(tiled.reshape(bsz*B, -1))

            # Initialize decoder input with start token for each beam
            flat_dec_input = np.full((bsz*B, 1), start_token, dtype='int32')

            # Initialize sequences and scores for each beam
            seqs   = [[[start_token]] * B for _ in range(bsz)]
            scores = np.zeros((bsz, B), dtype=np.float32)

            for t in range(max_dec_len):
                # Prepare inputs for decoder: current token and all states
                inputs = [flat_dec_input] + flat_states
                outs         = self.decoder_model.predict(inputs, verbose=0)
                logits       = outs[0]
                next_lp      = np.log(logits[:,0,:] + 1e-9)  # log-probabilities for next token
                next_lp      = next_lp.reshape(bsz, B, -1)

                init_states  = flat_states
                state_outputs= outs[1:len(init_states)+1]
                new_seqs     = []
                new_scores   = []
                new_states   = [np.zeros_like(s) for s in init_states]

                for i in range(bsz):
                    # For each item in batch, compute new beam candidates
                    total_lp    = scores[i][:,None] + next_lp[i]
                    flat_idx    = total_lp.reshape(-1)
                    topk_idx    = np.argpartition(-flat_idx, B-1)[:B]
                    topk_scores = flat_idx[topk_idx]
                    prev_beam   = topk_idx // next_lp.shape[2]
                    token_id    = topk_idx %  next_lp.shape[2]

                    bs_seqs = []
                    for j, (bprev, tok) in enumerate(zip(prev_beam, token_id)):
                        # Extend previous sequence with new token
                        bs_seqs.append(seqs[i][bprev] + [int(tok)])
                        src = i*B + bprev
                        dst = i*B + j
                        # Update decoder states for new beam
                        for k, st in enumerate(state_outputs):
                            new_states[k][dst] = st[src]

                    new_seqs.append(bs_seqs)
                    new_scores.append(topk_scores)

                # Update for next time step
                seqs        = new_seqs
                scores      = np.stack(new_scores, axis=0)
                flat_states = [s.reshape(bsz*B, -1) for s in new_states]
                # Prepare next decoder input (last token of each beam)
                flat_dec_input = np.array([[s[-1] for s in bs] for bs in seqs]).reshape(-1,1)

                # Early stopping if all beams in all batches ended
                if all(s[-1] == end_token for bs in seqs for s in bs):
                    break

            batch_preds = []
            # For each batch item, pick the best beam, remove start token, trim/pad to max_dec_len
            for bs in seqs:
                best_idx = int(np.argmax([scores[i,j] for j in range(B)]))
                seq = bs[best_idx]
                if seq and seq[0] == start_token:
                    seq = seq[1:]
                if end_token in seq:
                    seq = seq[:seq.index(end_token)+1]
                seq += [0] * (max_dec_len - len(seq))
                batch_preds.append(seq)

            # Compare predictions to targets for accuracy
            tgt_slice = target_seqs[bi*batch_size : bi*batch_size+bsz]
            for p, t in zip(batch_preds, tgt_slice):
                if np.array_equal(p, t):
                    total_correct += 1
            
            predictions.extend(batch_preds)

        return total_correct / N, predictions

class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        # Dense layers for computing attention scores
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)

    def call(self, query, values, mask=None):
        # Expand query and values for broadcasting in attention calculation
        q_expanded = tf.expand_dims(query, 2)   # (batch, T_dec, 1, hidden)
        v_expanded = tf.expand_dims(values, 1)  # (batch, 1, T_enc, hidden)

        # Compute attention scores (energy) using additive attention
        score = self.V(tf.nn.tanh(self.W1(q_expanded) + self.W2(v_expanded)))

        # Optionally mask out padding positions in encoder
        if mask is not None and mask[1] is not None:
            enc_mask = tf.expand_dims(mask[1], 1)
            score -= (1.0 - tf.cast(enc_mask, score.dtype)) * 1e9

        # Softmax over encoder time axis to get attention weights
        attn_weights = tf.nn.softmax(score, axis=2)
        attn_weights = tf.squeeze(attn_weights, -1)  # (batch, T_dec, T_enc)
        # Weighted sum of encoder outputs (context vector)
        context = tf.matmul(attn_weights, values)

        return context, attn_weights

class Seq2SeqAttention:
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim,
        hidden_units,
        dropout_rate=0.0,
        recurrent_dropout_rate=0.0,
        encoder_type='LSTM',
        decoder_type='LSTM',
        beam_width = 1
    ):
        # Store model hyperparameters
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.beam_width = beam_width

    def build_training_model(self):
        # Encoder input and embedding
        enc_inputs = Input(shape=(None,), name='encoder_inputs')
        enc_emb = Embedding(
            self.input_vocab_size,
            self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(enc_inputs)

        # Encoder RNN (LSTM/GRU/RNN)
        EncoderCell = getattr(tf.keras.layers, self.encoder_type)
        self.encoder_rnn = EncoderCell(
            self.hidden_units,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            name='encoder_'+self.encoder_type.lower()
        )
        enc_outputs_and_states = self.encoder_rnn(enc_emb)
        enc_outputs, *enc_states = enc_outputs_and_states

        # Decoder input and embedding
        dec_inputs = Input(shape=(None,), name='decoder_inputs')
        dec_emb = Embedding(
            self.output_vocab_size,
            self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(dec_inputs)

        # Decoder RNN (LSTM/GRU/RNN), initialized with encoder states
        DecoderCell = getattr(tf.keras.layers, self.decoder_type)
        self.decoder_rnn = DecoderCell(
            self.hidden_units,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            name='decoder_'+self.decoder_type.lower()
        )
        dec_outputs_and_states = self.decoder_rnn(
            dec_emb, initial_state=enc_states
        )
        dec_outputs, *dec_states = dec_outputs_and_states

        # Bahdanau attention layer
        self.attention_layer = BahdanauAttention(
            self.hidden_units, name='bahdanau_attn'
        )
        context, _ = self.attention_layer(
            dec_outputs, enc_outputs
        )

        # Concatenate decoder outputs and context vector
        concat = Concatenate(axis=-1, name='concat_layer')([dec_outputs, context])
        # Output layer: project to vocabulary size
        dec_logits = Dense(
            self.output_vocab_size,
            activation='softmax',
            name='output_dense'
        )(concat)

        # Define the full training model
        self.training_model = Model(
            inputs=[enc_inputs, dec_inputs],
            outputs=dec_logits,
            name='seq2seq_training'
        )
        
    def build_inference_model(self):
        # Encoder inference model for prediction
        enc_inputs_inf = Input(
            shape=(None,), name='encoder_inputs_inf'
        )
        enc_emb_inf = self.training_model.get_layer('encoder_embedding')(
            enc_inputs_inf
        )
        enc_rnn = self.training_model.get_layer(
            'encoder_'+self.encoder_type.lower()
        )
        enc_outputs_and_states = enc_rnn(enc_emb_inf)
        enc_outputs_inf, *enc_states_inf = enc_outputs_and_states

        self.encoder_model = Model(
            inputs=enc_inputs_inf,
            outputs=[enc_outputs_inf] + enc_states_inf,
            name='encoder_inference'
        )

        # Decoder inference model for step-by-step prediction
        dec_token_inf   = Input(shape=(1,), name='decoder_token_inf')
        enc_outputs_inp = Input(
            shape=(None, self.hidden_units),
            name='encoder_outputs_inf'
        )

        # Decoder state inputs for each state (h, c)
        dec_state_inputs = [
            Input(shape=(self.hidden_units,), name=f'decoder_state_inf_{i}')
            for i in range(len(enc_states_inf))
        ]

        dec_emb_inf = self.training_model.get_layer('decoder_embedding')(
            dec_token_inf
        )
        dec_rnn = self.training_model.get_layer(
            'decoder_'+self.decoder_type.lower()
        )
        dec_outputs_and_states_inf = dec_rnn(
            dec_emb_inf, initial_state=dec_state_inputs
        )
        dec_out_step, *dec_states_out = dec_outputs_and_states_inf

        # Compute context and output for this step
        context_inf, _ = self.attention_layer(
            dec_out_step, enc_outputs_inp
        )
        concat_inf = Concatenate(axis=-1)([dec_out_step, context_inf])
        dec_logits_inf = self.training_model.get_layer('output_dense')(
            concat_inf
        )

        # Final decoder inference model
        self.decoder_model = Model(
            inputs=[dec_token_inf, enc_outputs_inp] + dec_state_inputs,
            outputs=[dec_logits_inf] + dec_states_out,
            name='decoder_inference'
        )

    def build_attention_extractor(self):
        # Build a decoder model that also outputs attention weights for visualization
        dec_token_inf   = Input(shape=(1,), name='decoder_token_inf_attn')
        enc_outputs_inp = Input(
            shape=(None, self.hidden_units),
            name='encoder_outputs_inf_attn'
        )
        dec_state_inputs = [
            Input(shape=(self.hidden_units,), name=f'decoder_state_inf_attn_{i}')
            for i in range(len(self.encoder_model.output) - 1)
        ]
        dec_emb_inf = self.training_model.get_layer('decoder_embedding')(dec_token_inf)
        dec_rnn = self.training_model.get_layer('decoder_'+self.decoder_type.lower())
        dec_outputs_and_states_inf = dec_rnn(dec_emb_inf, initial_state=dec_state_inputs)
        dec_out_step, *dec_states_out = dec_outputs_and_states_inf

        # Get attention weights for this step
        context_inf, attn_weights = self.attention_layer(dec_out_step, enc_outputs_inp)
        concat_inf = Concatenate(axis=-1)([dec_out_step, context_inf])
        dec_logits_inf = self.training_model.get_layer('output_dense')(concat_inf)

        # Decoder model for extracting attention weights
        self.decoder_attn_model = Model(
            inputs=[dec_token_inf, enc_outputs_inp] + dec_state_inputs,
            outputs=[dec_logits_inf, attn_weights] + dec_states_out,
            name='decoder_inference_attn'
        )

    def compile(self, optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy']):
        # Compile the training model with optimizer, loss, and metrics
        self.training_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, x, y, batch_size=64, epochs=10, validation_split=0):
        # Train the model on the provided data
        self.training_model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
        )
    
    def evaluate(
        self,
        input_seqs,
        target_seqs,
        start_token,
        end_token,
        max_dec_len,
        batch_size=64
    ):
        """
        Batched beam search decoding + exact match accuracy.
        Uses one big GPU call per timestep over all (batch beam) hypotheses.
        """
        N = input_seqs.shape[0]
        n_batches = math.ceil(N / batch_size)
        total_correct = 0
        predictions = []

        for bi in range(n_batches):
            batch_inputs = input_seqs[bi*batch_size : (bi+1)*batch_size]
            bsz = batch_inputs.shape[0]

            # Encoder returns [encoder_outputs, state_h, state_c] (for LSTM)
            enc_results = self.encoder_model.predict(batch_inputs, verbose=0)
            enc_outputs = enc_results[0]
            enc_states = enc_results[1:]

            B = self.beam_width
            # Tile encoder outputs and states for beam search
            flat_enc_outputs = np.repeat(enc_outputs[:, None, :, :], B, axis=1).reshape(bsz*B, enc_outputs.shape[1], enc_outputs.shape[2])
            flat_states = []
            for state in enc_states:
                tiled = np.repeat(state[:, None, :], B, axis=1)
                flat_states.append(tiled.reshape(bsz*B, -1))

            flat_dec_input = np.full((bsz*B, 1), start_token, dtype='int32')

            seqs   = [[[start_token]] * B for _ in range(bsz)]
            scores = np.zeros((bsz, B), dtype=np.float32)

            for t in range(max_dec_len):
                # Decoder expects: [dec_token, enc_outputs, *states]
                inputs = [flat_dec_input, flat_enc_outputs] + flat_states
                outs   = self.decoder_model.predict(inputs, verbose=0)
                logits = outs[0]
                next_lp = np.log(logits[:,0,:] + 1e-9)

                next_lp = next_lp.reshape(bsz, B, -1)

                new_seqs  = []
                new_scores = []
                new_states = [np.zeros_like(s) for s in flat_states]

                for i in range(bsz):
                    total_lp = scores[i][:, None] + next_lp[i]
                    flat_indices = total_lp.reshape(-1)

                    topk_idx = np.argpartition(-flat_indices, B-1)[:B]
                    topk_scores = flat_indices[topk_idx]

                    prev_beam = topk_idx // next_lp.shape[2]
                    token_id  = topk_idx %  next_lp.shape[2]

                    bs_seqs = []
                    for j, (bprev, tok) in enumerate(zip(prev_beam, token_id)):
                        # Extend previous sequence with new token
                        seq = seqs[i][bprev] + [int(tok)]
                        bs_seqs.append(seq)

                        src_idx = i*B + bprev
                        dst_idx = i*B + j
                        # Update decoder states for new beam
                        for k, st in enumerate(outs[1:]):
                            new_states[k][dst_idx] = st[src_idx]

                    new_seqs.append(bs_seqs)
                    new_scores.append(topk_scores)

                seqs   = new_seqs
                scores = np.stack(new_scores, axis=0)
                flat_states = [ns.reshape(bsz*B, -1) for ns in new_states]
                last_tokens = [ [s[-1] for s in bs] for bs in seqs ]
                flat_dec_input = np.array(last_tokens).reshape(-1,1)

                # Early stopping if all beams in all batches ended
                if all(s[-1] == end_token for bs in seqs for s in bs):
                    break

            batch_preds = []
            for i, bs in enumerate(seqs):
                best_idx = int(np.argmax(scores[i]))
                seq = bs[best_idx]
                if seq and seq[0] == start_token:
                    seq = seq[1:]
                # Remove start token, trim at end token, pad to max_dec_len
                if end_token in seq:
                    seq = seq[:seq.index(end_token)+1]
                seq += [0] * (max_dec_len - len(seq))
                batch_preds.append(seq)

            tgt_slice = target_seqs[bi*batch_size : bi*batch_size+bsz]
            for p, t in zip(batch_preds, tgt_slice):
                if np.array_equal(p, t):
                    total_correct += 1

            predictions.extend(batch_preds)

        return total_correct / N, predictions

    def get_attention_for_example(self, input_seq, start_token, end_token, max_dec_len):
        # Encode input sequence to get encoder outputs and states
        enc_results = self.encoder_model.predict(input_seq[None, :], verbose=0)
        enc_outputs = enc_results[0]
        enc_states = enc_results[1:]

        dec_input = np.array([[start_token]])
        states = [s for s in enc_states]
        attn_matrices = []

        output_seq = []
        for _ in range(max_dec_len):
            # Run one decoding step and get attention weights
            outs = self.decoder_attn_model.predict([dec_input, enc_outputs] + states, verbose=0)
            logits, attn_weights, *states = outs
            pred_token = int(np.argmax(logits[0, 0]))
            output_seq.append(pred_token)
            attn_matrices.append(attn_weights[0])  # shape: (1, input_len)
            if pred_token == end_token:
                break
            dec_input = np.array([[pred_token]])

        # Trim output_seq and attn_matrices at first 0 (padding)
        if 0 in output_seq:
            idx = output_seq.index(0)
            output_seq = output_seq[:idx]
            attn_matrices = attn_matrices[:idx]
            
        attn_matrices = np.stack([np.squeeze(a) for a in attn_matrices], axis=0)

        # Trim attention columns for padded input tokens
        if isinstance(input_seq, np.ndarray):
            input_seq = input_seq.tolist()
        input_len = input_seq.index(0) if 0 in input_seq else len(input_seq)
        attn_matrices = attn_matrices[:, :input_len]

        return output_seq, attn_matrices
    
def cstr(s, color='black'):
    if s == ' ':
        return f"<span style='color:#000;padding-left:10px;background-color:{color}'>&nbsp;</span>"
    else:
        return f"<span style='color:#000;background-color:{color}'>{s} </span>"

def print_color(t):
    return ''.join([cstr(ti, color=ci) for ti,ci in t])

def get_clr(value):
    colors = [ '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#f9e8e8', '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f', '#f68f8f', '#f47676', '#f45f5f', '#f45f5f', '#f34343', '#f34343', '#f33b3b', '#f33b3b', '#f33b3b', '#f42e2e', '#f42e2e']
    value = int((value * 100) / 5)
    value = min(value, len(colors)-1)
    return colors[value]

def load_attention_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Load all three attention files
attn_files = {
    "LSTM": "attention_data_LSTM.jsonl",
    "GRU": "attention_data_GRU.jsonl",
    "SimpleRNN": "attention_data_SimpleRNN.jsonl"
}
attn_data = {k: load_attention_file(v) for k, v in attn_files.items()}


def get_by_idx(data, idx):
    for item in data:
        if item["idx"] == idx:
            return item
    return None

def cstr_hover(s, color='black', hover_text=None):
    # Use title attribute for hover
    hover = f" title='{hover_text}'" if hover_text else ""
    return f"<span style='color:#000;background-color:{color};padding:2px 6px;border-radius:3px;margin:1px;cursor:pointer;' {hover}>{s}</span>"

def print_color_hover(t):
    # t: list of (char, color, hover_text)
    return ''.join([cstr_hover(ti, color=ci, hover_text=ht) for ti,ci,ht in t])

def generate_html_for_step(items, step):
    html = "<div style='display:flex;gap:40px;margin-bottom:40px;'>"
    for name in attn_files.keys():
        if name not in items:
            html += f"<div style='min-width:250px;'><b>{name}</b>: Not found</div>"
            continue
        attn_weights = items[name]["attn_weights"]
        outputs = items[name]["outputs"]
        input_labels = items[name]["input_labels"]
        if step >= len(attn_weights):
            html += f"<div style='min-width:250px;'><b>{name}</b>: Done</div>"
            continue
        # Input row (no hover)
        text_ce = []
        for j in range(len(input_labels)):
            text_e = (input_labels[j], get_clr(attn_weights[step][j]), None)
            text_ce.append(text_e)
        # Output row (hover shows distribution)
        text_he = []
        for j in range(len(outputs)):
            # Only highlight the output at this step, others are gray/white
            if j == step and j < len(attn_weights) and j < len(attn_weights[step]):
                color = get_clr(attn_weights[step][j])
            else:
                color = get_clr(0)
            # Prepare hover text: show attention distribution for this output step
            if j < len(attn_weights):
                dist = ", ".join(f"{input_labels[k]}: {attn_weights[j][k]:.2f}" for k in range(len(input_labels)))
            else:
                dist = ""
            text_h = (outputs[j], color, dist)
            text_he.append(text_h)
        block = f"<b>{name}</b><br>"
        block += print_color_hover(text_he) + "<br>" + print_color_hover(text_ce)
        html += f"<div style='min-width:250px;'>{block}</div>"
    html += "</div>"
    return html

def export_parallel_visualization_html(idx, out_path="attention_comparison.html"):
    items = {}
    max_steps = 0
    for name, data in attn_data.items():
        item = get_by_idx(data, idx)
        if item is None:
            continue
        attn_weights = np.array(item["attention"])
        outputs = item["output_labels"]
        input_labels = item["input_labels"]
        if outputs:
            outputs = outputs[:-1] + ['<e>']
        attn_weights = attn_weights[:-1]
        for i in range(len(attn_weights)):
            attn_weights[i] = attn_weights[i][:len(input_labels)]
        attn_weights = np.asarray(attn_weights)
        items[name] = {
            "attn_weights": attn_weights,
            "outputs": outputs,
            "input_labels": input_labels
        }
        max_steps = max(max_steps, len(attn_weights))

    html = """
    <html>
    <head>
    <meta charset='utf-8'>
    <title>Hover Over the Character to see the distribution of attention from Input</title>
    <style>
    body { font-family: 'Nirmala UI', 'Gujarati Saral', 'Mangal', sans-serif; }
    .attn-row { display: flex; gap: 40px; margin-bottom: 40px; }
    .attn-box { min-width: 250px; }
    .attn-title { font-weight: bold; font-size: 1.1em; margin-bottom: 6px; }
    .attn-output { font-size: 1.2em; margin-bottom: 6px; }
    .attn-input { font-size: 1.1em; margin-bottom: 6px; }
    .attn-bar { margin-top: 10px; }
    .attn-cell { display: inline-block; padding: 2px 6px; border-radius: 3px; margin: 1px; min-width: 24px; text-align: center; }
    .step-section { display: none; }
    </style>
    <script>
    function showAttn(id) {
        var bars = document.getElementsByClassName('attn-bar');
        for (var i = 0; i < bars.length; ++i) bars[i].style.display = 'none';
        var el = document.getElementById(id);
        if (el) el.style.display = 'block';
    }
    function hideAttn(id) {
        var el = document.getElementById(id);
        if (el) el.style.display = 'none';
    }
    var currentStep = 0;
    function showStep(idx, total) {
        for (var i = 0; i < total; ++i) {
            var sec = document.getElementById('step_section_' + i);
            if (sec) sec.style.display = (i == idx ? 'block' : 'none');
        }
        document.getElementById('step_num').innerText = (idx+1) + ' / ' + total;
        currentStep = idx;
    }
    function nextStep(total) {
        if (currentStep < total-1) showStep(currentStep+1, total);
    }
    function prevStep(total) {
        if (currentStep > 0) showStep(currentStep-1, total);
    }
    </script>
    </head>
    <body>
    <h2>Attention Visualization Comparison</h2>
    <div>
      <button onclick="prevStep({max_steps})">&lt; Prev</button>
      <span id="step_num">1 / {max_steps}</span>
      <button onclick="nextStep({max_steps})">Next &gt;</button>
    </div>
    """.replace("{max_steps}", str(max_steps))

    for step in range(max_steps):
        html += f"<div class='step-section' id='step_section_{step}' style='display:{'block' if step==0 else 'none'};'>"
        html += f"<h3>Step {step+1}</h3>"
        html += "<div class='attn-row'>"
        for name in attn_files.keys():
            if name not in items:
                html += f"<div class='attn-box'><span class='attn-title'>{name}</span><br>Not found</div>"
                continue
            attn_weights = items[name]["attn_weights"]
            outputs = items[name]["outputs"]
            input_labels = items[name]["input_labels"]
            if step >= len(attn_weights):
                html += f"<div class='attn-box'><span class='attn-title'>{name}</span><br>Done</div>"
                continue

            # Output row with hover
            output_html = ""
            attn_bars_html = ""
            for j, out_char in enumerate(outputs):
                attn_id = f"attn_{name}_{step}_{j}"
                if j < len(attn_weights):
                    bar = "<div class='attn-bar' id='{0}' style='display:none;'>".format(attn_id)
                    for k, in_char in enumerate(input_labels):
                        color = get_clr(attn_weights[j][k])
                        val = attn_weights[j][k]
                        bar += f"<span class='attn-cell' style='background:{color}' title='{in_char}: {val:.2f}'>{in_char}<br><span style='font-size:0.8em'>{val:.2f}</span></span>"
                    bar += "</div>"
                    attn_bars_html += bar
                    if j == step and j < attn_weights.shape[1]:
                        bg_color = get_clr(attn_weights[step][j])
                    else:
                        bg_color = get_clr(0)
                    output_html += f"<span class='attn-cell' style='background:{bg_color};cursor:pointer;' onmouseover=\"showAttn('{attn_id}')\" onmouseout=\"hideAttn('{attn_id}')\">{out_char}</span>"
                else:
                    output_html += f"<span class='attn-cell'>{out_char}</span>"
            input_html = ""
            for k, in_char in enumerate(input_labels):
                input_html += f"<span class='attn-cell'>{in_char}</span>"

            html += f"<div class='attn-box'><div class='attn-title'>{name}</div>"
            html += f"<div class='attn-output'>{output_html}</div>"
            html += f"<div class='attn-input'>{input_html}</div>"
            html += attn_bars_html
            html += "</div>"
        html += "</div></div>"
    html += """
    <script>showStep(0, {max_steps});</script>
    </body></html>
    """.replace("{max_steps}", str(max_steps))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def plot_decoder_activations_multi_combined(json_path, neuron_idx=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all neuron indices
    all_neurons = set()
    for example in data:
        all_neurons.update(example["neuron_indices"])
    if neuron_idx is not None:
        all_neurons = set([neuron_idx])

    for nidx in sorted(all_neurons):
        # Filter examples that have this neuron
        filtered = [ex for ex in data if str(nidx) in ex["activations"]]
        if not filtered:
            continue
        n_examples = len(filtered)
        # Use fixed cell size: width = cell_width * max_T_dec, height = cell_height * n_examples
        cell_width = 0.7
        cell_height = 1.2
        max_T_dec = max(len(ex["decoded_chars"]) for ex in filtered)
        fig, axes = plt.subplots(
            n_examples, 1, figsize=(cell_width * max_T_dec, cell_height * n_examples),
            squeeze=False
        )
        fig.suptitle(f"Neuron {nidx} activations across {n_examples} examples", fontsize=16)
        for i, example in enumerate(filtered):
            decoded_chars = example["decoded_chars"]
            acts = example["activations"][str(nidx)]
            abs_vals = np.abs(acts)
            vmin = 0
            vmax = max(abs_vals)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.RdBu
            ax = axes[i, 0]
            ax.axis('off')
            for j, (ch, act) in enumerate(zip(decoded_chars, acts)):
                facecolor = cmap(norm(abs(act)))
                rect = mpl.patches.Rectangle((j, 0), 1, 1, facecolor=facecolor, edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
                ax.text(
                    j + 0.5, 0.5, ch,
                    fontsize=16,
                    ha='center', va='center',
                    color='black'  # Always black
                )
            ax.set_xlim(0, max_T_dec)
            ax.set_ylim(0, 1)
            ax.set_title(f"Example idx={example['idx']}", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()