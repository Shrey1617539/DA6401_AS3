import numpy
import tensorflow.keras as keras

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
        encoder_inputs = self.training_model.get_layer('encoder_inputs').input
        encoder_embedding = self.training_model.get_layer('encoder_embedding')(encoder_inputs)

        encoder_output = encoder_embedding
        encoder_states = []
        for i in range(self.encoder_layers):
            layer = self.training_model.get_layer('encoder_{}'.format(i))
            encoder_output, *states = layer(encoder_output)
            encoder_states.extend(states)


        # if self.encoder_type == 'LSTM':
        #     encoder_states = [
        #         self.training_model.get_layer('encoder_{}'.format(self.encoder_layers - 1)).output[0],
        #         self.training_model.get_layer('encoder_{}'.format(self.encoder_layers - 1)).output[1]
        #     ]
        # else:
        #     encoder_states = [
        #         self.training_model.get_layer('encoder_{}'.format(self.encoder_layers - 1)).output
        #     ]
        
        self.encoder_model = keras.models.Model(
            inputs=encoder_inputs,
            outputs=encoder_states
        )



        decoder_inputs = self.training_model.get_layer('decoder_inputs').input
        decoder_embedding = self.training_model.get_layer('decoder_embedding')(decoder_inputs)
        
        decoder_states_inputs = []
        for idx, state in enumerate(encoder_states):
            decoder_states_inputs.append(
                keras.layers.Input(shape=(self.hidden_units,), name=f'decoder_state_input_{idx}')
            )
        # if self.decoder_type == 'LSTM':
        #     decoder_states_inputs = keras.layers.Input(shape=(self.hidden_units,))
        # else:
        #     decoder_state_input = keras.layers.Input(shape=(self.hidden_units,))
        #     decoder_states_inputs = [decoder_state_input]
        

        decoder_outputs = decoder_embedding
        decoder_states = []
        state_idx = 0
        for i in range(self.decoder_layers):
            layer = self.training_model.get_layer('decoder_{}'.format(i))
            if i < self.encoder_layers:
                if self.decoder_type == 'LSTM':
                    init_h = decoder_states_inputs[state_idx]
                    init_c = decoder_states_inputs[state_idx + 1]
                    decoder_outputs, state_h, state_c = layer(
                        decoder_outputs,
                        initial_state=[init_h, init_c]
                    )
                    decoder_states.extend([state_h, state_c])
                    state_idx += 2
                else:
                    init_h = decoder_states_inputs[state_idx]
                    decoder_outputs, state_h = layer(
                        decoder_outputs,
                        initial_state=[init_h]
                    )
                    decoder_states.append(state_h)
                    state_idx += 1
            else:
                if self.decoder_type == 'LSTM':
                    init_h = decoder_states_inputs[-2]
                    init_c = decoder_states_inputs[-1]
                    decoder_outputs, state_h, state_c = layer(
                        decoder_outputs,
                        initial_state=[init_h, init_c]
                    )
                    decoder_states.extend([state_h, state_c])
                else:
                    init_h = decoder_states_inputs[-1]
                    decoder_outputs, state_h = layer(
                        decoder_outputs,
                        initial_state=[init_h]
                    )
                    decoder_states.append(state_h)

            # outputs = layer(decoder_outputs, initial_state=decoder_states_inputs)
    
            
            # if self.decoder_type == 'LSTM':
            #     decoder_outputs, state_h, state_c = outputs
            #     decoder_states = [state_h, state_c]
            # else:
            #     decoder_outputs, state_h = outputs
                # decoder_states = [state_h]
        
        decoder_outputs = self.training_model.get_layer('decoder_dense')(decoder_outputs)

        self.decoder_model = keras.models.Model(
            inputs=[decoder_inputs] + decoder_states_inputs,
            outputs=[decoder_outputs] + decoder_states
        )
        # for i in range(self.decoder_layers):
        #     if self.decoder_type == 'LSTM':
        #         decoder_lstm = self.training_model.get_layer('decoder_{}'.format(i))

        #         if i == 0:
        #             decoder_outputs, state_h, state_c = decoder_lstm(
        #                 decoder_outputs,
        #                 initial_state=decoder_states_inputs
        #             )
        #             decoder_states = [state_h, state_c]
        #         else:
        #             decoder_outputs = decoder_lstm(decoder_outputs)
            
        #     elif self.decoder_type == 'GRU':
        #         decoder_gru = self.training_model.get_layer('decoder_{}'.format(i))

        #         if i == 0:
        #             decoder_outputs, state_h = decoder_gru(
        #                 decoder_outputs,
        #                 initial_state=decoder_states_inputs
        #             )
        #             decoder_states = [state_h]
        #         else:
        #             decoder_outputs = decoder_gru(decoder_outputs)
            
        #     elif self.decoder_type == 'RNN':
        #         decoder_rnn = self.training_model.get_layer('decoder_{}'.format(i))

        #         if i == 0:
        #             decoder_outputs, state_h = decoder_rnn(
        #                 decoder_outputs,
        #                 initial_state=decoder_states_inputs
        #             )
        #             decoder_states = [state_h]
        #         else:
        #             decoder_outputs = decoder_rnn(decoder_outputs)
        
        # decoder_dense = self.training_model.get_layer('decoder_dense')
        # decoder_outputs = decoder_dense(decoder_outputs)
        # self.decoder_model = keras.models.Model(
        #     inputs=[decoder_inputs] + decoder_states_inputs,
        #     outputs=[decoder_outputs] + decoder_states
        # )
                    
