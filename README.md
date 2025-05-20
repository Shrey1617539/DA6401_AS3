# DA6401 AS3

A deep learning project implementing sequence-to-sequence models with and without attention mechanisms for neural machine translation tasks. The project focuses on character-level transliteration between languages using various RNN architectures.

## Links

## Overview

This project implements neural machine translation using encoder-decoder architectures with and without attention mechanisms. It supports different types of recurrent neural networks (LSTM, GRU, and SimpleRNN) and includes tools for visualizing attention weights to interpret the model's behavior. The implementation is based on TensorFlow/Keras and includes integration with Weights & Biases for experiment tracking.

## Features

- **Multiple Seq2Seq Architectures**: Vanilla encoder-decoder models, Encoder-decoder models with attention mechanism

- **Flexible RNN Cell Types**: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit), SimpleRNN

- **Advanced Decoding**: Beam search decoding for improved translation quality

- **Visualization Tools**: Attention weight visualization, Cell activation visualization for LSTM models

- **Experiment Tracking**: Integration with Weights & Biases for monitoring training progress

- **Customizable Training**: Extensive hyperparameter configuration via command-line arguments

## File Structure

```
DA6401_AS3/
├── predictions_attention/       # Predictions from attention-based models
│   └── predictions.csv
├── predictions_vanilla/         # Predictions from vanilla models
│   └── predictions.csv
├── attention_data_GRU.jsonl     # Attention visualization data for GRU
├── attention_data_LSTM.jsonl    # Attention visualization data for LSTM
├── attention_data_SimpleRNN.jsonl # Attention visualization data for SimpleRNN
├── DA6401_AS3.ipynb            # Jupyter notebook for the assignment
├── helping_functions.py        # Utility functions and model implementations
└── train.py                    # Main training script
```

## Requirements

- Python 3.6+

- TensorFlow 2.x

- Keras

- NumPy

- Pandas

- Matplotlib

- Weights & Biases (wandb)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Shrey1617539/DA6401_AS3
```

2. **Navigate to the project directory:**

```bash
cd DA6401_AS3
```

## USAGE

### Training a Model

- **Run with default hyperparameters:**

```
python train.py
```

- **Run with custom settings:**

```
python train.py --wandb_entity your_entity --wandb_project your_project --dataset_train /path/to/train --dataset_test /path/to/test --epochs 20 --batch_size 64 --learning_rate 0.001
```

This will launch a training session with specified parameters and log metrics to wandb.

### Running the Notebook

Open the Q1.ipynb notebook with Jupyter. The notebook provides an alternative interface to run experiments, including configuring and launching hyperparameter sweeps via wandb.

## Command Line Arguments

### `train.py` Arguments

| Argument            | Flag(s)                               | Type    | Default                   | Description                                   |
| ------------------- | ------------------------------------- | ------- | ------------------------- | --------------------------------------------- |
| Wandb Entity        | `-we`, `--wandb_entity`               | `str`   | `None`                    | Weights & Biases entity name                  |
| Wandb Project       | `-wp`, `--wandb_project`              | `str`   | `None`                    | Project name for experiment tracking          |
| Dataset             | `-d`, `--dataset`                     | `str`   | `"path/to/training/data"` | Directory location of dataset                 |
| Cell Type           | `-ct`, `--cell_type`                  | `str`   | `"GRU"`                   | Type of RNN cell (`LSTM`, `GRU`, `SimpleRNN`) |
| Attention           | `-att`, `--attention`                 | `bool`  | `True`                    | Whether to use attention mechanism            |
| Attention Extractor | `-att_ex`, `--attention_extractor`    | `bool`  | `False`                   | Extract attention weights for visualization   |
| Connectivity        | `-conn`, `--connectivity`             | `bool`  | `False`                   | Generate connectivity visualization           |
| Batch Size          | `-bs`, `--batch_size`                 | `int`   | `400`                     | Batch size for training                       |
| Epochs              | `-e`, `--epochs`                      | `int`   | `10`                      | Number of training epochs                     |
| Learning Rate       | `-lr`, `--learning_rate`              | `float` | `0.0041760877805365965`   | Learning rate for optimizer                   |
| Embedding Dimension | `-emb`, `--embedding_dim`             | `int`   | `32`                      | Dimension of embedding layers                 |
| Hidden Units        | `-hu`, `--hidden_units`               | `int`   | `256`                     | Number of units in RNN layers                 |
| Dropout Rate        | `-dr`, `--dropout_rate`               | `float` | `0.0640614879476808`      | Dropout rate for regularization               |
| Recurrent Dropout   | `-rec_dr`, `--recurrent_dropout_rate` | `float` | `0.14003195916675987`     | Recurrent dropout rate                        |
| Encoder Layers      | `-enc_layers`, `--encoder_layers`     | `int`   | `1`                       | Number of encoder layers                      |
| Decoder Layers      | `-dec_layers`, `--decoder_layers`     | `int`   | `1`                       | Number of decoder layers                      |
| Beam Width          | `-beam`, `--beam_width`               | `int`   | `3`                       | Width for beam search decoding                |
| Validation          | `-do_val`, `--do_val`                 | `bool`  | `False`                   | Evaluate on validation set                    |
| Testing             | `-do_test`, `--do_test`               | `bool`  | `False`                   | Evaluate on test set                          |
