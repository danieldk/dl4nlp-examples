from enum import Enum

class DefaultConfig:
    n_epochs = 50
    start_lr = 0.01
    decay_steps = 177
    decay_rate = 0.90

    batch_size = 512
    max_timesteps = 20

    char_embed_size = 40

    # Output sizes of the RNN layers.
    hidden_sizes = [250]

    # Character embedding dropout
    input_dropout = 0.9

    # RNN output dropout
    rnn_output_dropout = 0.9

    # RNN state dropout
    rnn_state_dropout = 0.9

