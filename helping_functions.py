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
    
    def build_model(self):
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
            return_state = (i == self.encoder_layers - 1)

            if self.encoder_type == 'LSTM':
                rnn_layer = keras.layers.LSTM(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'encoder_{i}'
                )

                if return_state:
                    encoder_outputs, state_h, state_c = rnn_layer(encoder_outputs)
                    encoder_states = [state_h, state_c]
                else:
                    encoder_outputs = rnn_layer(encoder_outputs)
            
            elif self.encoder_type == 'GRU':
                rnn_layer = keras.layers.GRU(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'encoder_{i}'
                )

                if return_state:
                    encoder_outputs, state_h = rnn_layer(encoder_outputs)
                    encoder_states = [state_h]
                else:
                    encoder_outputs = rnn_layer(encoder_outputs)
            
            elif self.encoder_type == 'RNN':
                rnn_layer = keras.layers.SimpleRNN(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'encoder_{i}'
                )

                if return_state:
                    encoder_outputs, state_h = rnn_layer(encoder_outputs)
                    encoder_states = [state_h]
                else:
                    encoder_outputs = rnn_layer(encoder_outputs)
        
        decoder_inputs = keras.layers.Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = keras.layers.Embedding(
            input_dim=self.output_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)

        decoder_outputs = decoder_embedding
        decoder_states = encoder_states

        for i in range(self.decoder_layers):
            return_sequences = True
            return_state = False

            if self.decoder_type == 'LSTM':
                rnn_layer = keras.layers.LSTM(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )
                if i == 0:
                    decoder_outputs = rnn_layer(decoder_outputs, initial_state=decoder_states)
                else:
                    decoder_outputs = rnn_layer(decoder_outputs)
                
            elif self.decoder_type == 'GRU':
                rnn_layer = keras.layers.GRU(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )

                if i == 0:
                    decoder_outputs = rnn_layer(decoder_outputs, initial_state=decoder_states)
                else:
                    decoder_outputs = rnn_layer(decoder_outputs)
            
            elif self.decoder_type == 'RNN':
                rnn_layer = keras.layers.SimpleRNN(
                    units=self.hidden_units,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'decoder_{i}'
                )

                if i == 0:
                    decoder_outputs = rnn_layer(decoder_outputs, initial_state=decoder_states)
                else:
                    decoder_outputs = rnn_layer(decoder_outputs)

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