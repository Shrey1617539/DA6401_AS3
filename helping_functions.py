import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
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
        encoder_inputs = keras.layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = keras.layers.Embedding(
            input_dim=self.input_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)

        encoder_states = []
        encoder_outputs = encoder_embedding

        for i in range(self.encoder_layers):
            return_sequences = (i < self.encoder_layers - 1)
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

            elif self.encoder_type == 'RNN':
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
        
        decoder_inputs = keras.layers.Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = keras.layers.Embedding(
            input_dim=self.output_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)

        decoder_outputs = decoder_embedding
        decoder_init_states = []
        idx = 0
        for i in range(self.decoder_layers):
            if i<self.encoder_layers:
                if self.decoder_type == 'LSTM':
                    h = encoder_states[idx]
                    c = encoder_states[idx + 1]
                    decoder_init_states.append([h,c])
                    idx += 2
                else:
                    h = encoder_states[idx]
                    decoder_init_states.append([h])
                    idx += 1
            else:
                if self.decoder_type == 'LSTM':
                    h = encoder_states[-2]
                    c = encoder_states[-1]
                    decoder_init_states.append([h,c])
                else:
                    h = encoder_states[-1]
                    decoder_init_states.append([h])

                

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

            
            elif self.decoder_type == 'RNN':
                rnn_layer = keras.layers.SimpleRNN(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )
                decoder_outputs, _ = rnn_layer(decoder_outputs, initial_state=decoder_init_states[i])


        decoder_dense = keras.layers.Dense(
            units=self.output_vocab_size,
            activation='softmax',
            name='decoder_dense'
        )
        decoder_outputs = decoder_dense(decoder_outputs)
        self.training_model = keras.models.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=decoder_outputs
        )
    
    def build_inference_model(self):
        # Encoder
        encoder_inputs = keras.layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding_layer = self.training_model.get_layer('encoder_embedding')
        encoder_embedding = encoder_embedding_layer(encoder_inputs)

        encoder_outputs = encoder_embedding
        encoder_states = []
        for i in range(self.encoder_layers):
            encoder_rnn_layer = self.training_model.get_layer(f'encoder_{i}')
            encoder_outputs, *state = encoder_rnn_layer(encoder_outputs)
            encoder_states.extend(state)

        self.encoder_model = keras.models.Model(
            inputs=encoder_inputs,
            outputs=encoder_states
        )

        # Decoder
        decoder_inputs = keras.layers.Input(shape=(None,), name='decoder_inputs')
        decoder_embedding_layer = self.training_model.get_layer('decoder_embedding')
        decoder_embedding = decoder_embedding_layer(decoder_inputs)

        decoder_states_inputs = []
        for idx, state in enumerate(encoder_states):
            decoder_states_inputs.append(
                keras.layers.Input(shape=(self.hidden_units,), name=f'decoder_state_input_{idx}')
            )

        decoder_outputs = decoder_embedding
        state_idx = 0
        decoder_states = []
        for i in range(self.decoder_layers):
            decoder_rnn_layer = self.training_model.get_layer(f'decoder_{i}')
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

        decoder_dense_layer = self.training_model.get_layer('decoder_dense')
        decoder_outputs = decoder_dense_layer(decoder_outputs)

        self.decoder_model = keras.models.Model(
            inputs=[decoder_inputs] + decoder_states_inputs,
            outputs=[decoder_outputs] + decoder_states
        )
    
    def compile(self, optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy']):
        self.training_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, x, y, batch_size=64, epochs=10, validation_split=0):
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

        for bi in range(n_batches):
            print(bi)
            # --- 1) Slice batch ---
            batch_inputs = input_seqs[bi*batch_size : (bi+1)*batch_size]
            bsz = batch_inputs.shape[0]

            # --- 2) Encoder: get initial states ---
            enc_states = self.encoder_model.predict(batch_inputs, verbose=0)
            # enc_states: list of arrays, each (bsz, hidden)

            # --- 3) Tile states & init beams ---
            B = self.beam_width
            # For each state array, expand (bsz,hidden)->(bsz,B,hidden) then flatten
            flat_states = []
            for state in enc_states:
                tiled = np.repeat(state[:, None, :], B, axis=1)
                flat_states.append(tiled.reshape(bsz*B, -1))

            # Each hypothesis starts with one <start> token
            flat_dec_input = np.full((bsz*B, 1), start_token, dtype='int32')

            # Keep track of:
            #   - sequences: list of token‐lists per (batch,beam)
            #   - scores: log‐probs per (batch,beam)
            seqs   = [[[start_token]] * B for _ in range(bsz)]
            scores = np.zeros((bsz, B), dtype=np.float32)

            # --- 4) Beam‐search loop ---
            for t in range(max_dec_len):
                # 4a) One big predict over all hypotheses
                inputs = [flat_dec_input] + flat_states
                outs   = self.decoder_model.predict(inputs, verbose=0)
                logits = outs[0]                # shape: (bsz*B, 1, V)
                next_lp = np.log(logits[:,0,:] + 1e-9)  # (bsz*B, V)

                # 4b) split back to (bsz, B, V)
                next_lp = next_lp.reshape(bsz, B, -1)

                new_seqs  = []
                new_scores = []
                new_states = [np.zeros_like(s) for s in flat_states]

                # 4c) For each item in batch, pick top B out of B×V
                for i in range(bsz):
                    # accumulate scores + new log‐probs
                    total_lp = scores[i][:, None] + next_lp[i]   # (B, V)
                    flat_indices = total_lp.reshape(-1)          # (B*V,)

                    # top‐B indices in flattened B*V
                    topk_idx = np.argpartition(-flat_indices, B-1)[:B]
                    topk_scores = flat_indices[topk_idx]

                    # decode which beam & token each came from
                    prev_beam = topk_idx // next_lp.shape[2]      # (B,)
                    token_id  = topk_idx %  next_lp.shape[2]      # (B,)

                    # build new sequences & gather new states
                    bs_seqs = []
                    for j, (bprev, tok) in enumerate(zip(prev_beam, token_id)):
                        # copy old sequence + new token
                        seq = seqs[i][bprev] + [int(tok)]
                        bs_seqs.append(seq)

                        # compute new state slice index in flat arrays
                        src_idx = i*B + bprev
                        dst_idx = i*B + j
                        # copy each state vector for this hypothesis
                        for k, st in enumerate(outs[1:]):
                            new_states[k][dst_idx] = st[src_idx]

                    new_seqs.append(bs_seqs)
                    new_scores.append(topk_scores)

                # 4d) reorganize for next step
                seqs   = new_seqs
                scores = np.stack(new_scores, axis=0)  # (bsz, B)
                flat_states = [ns.reshape(bsz*B, -1) for ns in new_states]
                # decoder input = last token of each hyp
                last_tokens = [ [s[-1] for s in bs] for bs in seqs ]  # list of lists
                flat_dec_input = np.array(last_tokens).reshape(-1,1)

                # 4e) check for all‐ended
                # if every top sequence ends in <end>, we can stop early
                if all(s[-1] == end_token for bs in seqs for s in bs):
                    break

            # --- 5) Select best beam & pad to max_dec_len ---
            batch_preds = []
            for bs in seqs:
                # pick beam with highest score
                best_idx = int(np.argmax([scores[i,j] for j in range(B)]))
                seq = bs[best_idx]
                # strip <start>, cut at <end>, then post‐pad 0
                seq = [tok for tok in seq if tok not in (start_token,)]
                if end_token in seq:
                    seq = seq[:seq.index(end_token)]
                seq += [0] * (max_dec_len - len(seq))
                batch_preds.append(seq)

            # --- 6) exact‐match for this slice ---
            tgt_slice = target_seqs[bi*batch_size : bi*batch_size+bsz]
            for p, t in zip(batch_preds, tgt_slice):
                if np.array_equal(p, t):
                    total_correct += 1

        return total_correct / N