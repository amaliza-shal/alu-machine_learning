#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    input_dims: tuple of dimensions of the model input
    filters: list of number of filters for each convolutional layer
    latent_dims: tuple of dimensions of the latent space representation
    Returns: encoder, decoder, auto
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    encoder = keras.Model(encoder_input, x)

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    # filters are reversed for the decoder [16, 8, 8] -> [8, 8, 16]
    # Except for the last two convolutions:
    # 1. Standard Conv + Upsampling for filters[:-1]
    # 2. Second to last: Valid padding
    # 3. Last: sigmoid + no upsampling
    
    rev_filters = filters[::-1]
    
    # Each convolution in the decoder, except for the last two
    for i in range(len(rev_filters) - 1):
        x = keras.layers.Conv2D(
            filters=rev_filters[i], kernel_size=(3, 3),
            padding='same', activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        
    # The second to last convolution should instead use valid padding
    x = keras.layers.Conv2D(
        filters=rev_filters[-1], kernel_size=(3, 3),
        padding='valid', activation='relu'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # The last convolution: same number of filters as channels in input_dims
    # sigmoid activation and no upsampling
    x = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=(3, 3),
        padding='same', activation='sigmoid'
    )(x)
    
    decoder = keras.Model(decoder_input, x)

    # Autoencoder
    auto_input = encoder_input
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
