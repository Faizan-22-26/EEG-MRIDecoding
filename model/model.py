import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Raw EEG Encoder
def raw_eeg_encoder(input_shape):
    raw_eeg_input = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(raw_eeg_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128)(x)
    raw_eeg_output = layers.Dense(128, activation='relu')(x)
    return models.Model(raw_eeg_input, raw_eeg_output, name='raw_eeg_encoder')

# Spectrogram Encoder
def spectrogram_encoder(input_shape):
    spectrogram_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spectrogram_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    spectrogram_output = layers.Dense(128, activation='relu')(x)
    return models.Model(spectrogram_input, spectrogram_output, name='spectrogram_encoder')

# Combined EEG Encoder
def eeg_encoder(raw_eeg_shape, spectrogram_shape):
    raw_eeg_enc = raw_eeg_encoder(raw_eeg_shape)
    spectrogram_enc = spectrogram_encoder(spectrogram_shape)
    combined_input = layers.concatenate([raw_eeg_enc.output, spectrogram_enc.output])
    shared_latent = layers.Dense(128, activation='relu')(combined_input)
    return models.Model([raw_eeg_enc.input, spectrogram_enc.input], shared_latent, name='eeg_encoder')

# MRI Encoder
def mri_encoder(input_shape):
    mri_input = layers.Input(shape=input_shape)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(mri_input)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling3D()(x)
    mri_output = layers.Dense(128, activation='relu')(x)
    return models.Model(mri_input, mri_output, name='mri_encoder')
def transformer_block(latent_dim, num_heads, ff_dim, dropout_rate=0.1):
    inputs = layers.Input(shape=(latent_dim,))
    reshaped = layers.Reshape((1, latent_dim))(inputs)  # Add sequence dimension
    x = layers.LayerNormalization(epsilon=1e-6)(reshaped)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=latent_dim)(x, x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Add()([reshaped, x])  # Residual connection

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x_ff = layers.Dense(ff_dim, activation='relu')(x)
    x_ff = layers.Dense(latent_dim)(x_ff)
    x_ff = layers.Dropout(dropout_rate)(x_ff)
    outputs = layers.Add()([x, x_ff])  # Residual connection
    outputs = layers.Reshape((latent_dim,))(outputs)  # Remove sequence dimension
    return models.Model(inputs, outputs, name='transformer_block')

def eeg_decoder(latent_dim, output_shape):
    latent_input = layers.Input(shape=(latent_dim,))


    reshaped_size = (output_shape[0] // 2, output_shape[1] // 2, 64)

    x = layers.Dense(np.prod(reshaped_size), activation='relu')(latent_input)
    x = layers.Reshape(reshaped_size)(x)


    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample to (output_shape[0], output_shape[1])
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    eeg_output = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(latent_input, eeg_output, name='eeg_decoder')
def mri_decoder(latent_dim, output_shape):
    latent_input = layers.Input(shape=(latent_dim,))


    reshaped_size = (output_shape[0] // 2, output_shape[1] // 2, output_shape[2] // 2, 64)

    x = layers.Dense(np.prod(reshaped_size), activation='relu')(latent_input)
    x = layers.Reshape(reshaped_size)(x)


    x = layers.Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    x = layers.Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(x)
    mri_output = layers.Conv3DTranspose(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(latent_input, mri_output, name='mri_decoder')

# Cross-Modality Model with Transformer in Latent Space
def cross_modality_model_with_transformer(raw_eeg_shape, spectrogram_shape, mri_shape):

    raw_eeg_enc = raw_eeg_encoder(raw_eeg_shape)
    spectrogram_enc = spectrogram_encoder(spectrogram_shape)
    eeg_combined_input = layers.concatenate([raw_eeg_enc.output, spectrogram_enc.output])
    eeg_shared_latent = layers.Dense(128, activation='relu')(eeg_combined_input)

    mri_enc = mri_encoder(mri_shape)
    mri_latent = mri_enc.output

    eeg_latent_input = layers.Input(shape=(128,))
    mri_latent_input = layers.Input(shape=(128,))
    transformer = transformer_block(128, num_heads=4, ff_dim=256)
    aligned_latent_eeg = transformer(eeg_latent_input)
    aligned_latent_mri = transformer(mri_latent_input)



    eeg_decoder_model = eeg_decoder(latent_dim=128, output_shape=spectrogram_shape)
    mri_decoder_model = mri_decoder(latent_dim=128, output_shape=mri_shape)

    eeg_reconstruction = eeg_decoder_model(aligned_latent_eeg)
    mri_reconstruction = mri_decoder_model(aligned_latent_mri)

    # Hopfield-like retrieval
    hopfield_mapping = layers.Dense(128, activation='relu')(aligned_latent_eeg)
    hopfield_mri_reconstruction = mri_decoder_model(hopfield_mapping)


    model = models.Model(
        inputs=[raw_eeg_enc.input, spectrogram_enc.input, mri_enc.input],
        outputs=[eeg_reconstruction, mri_reconstruction, hopfield_mri_reconstruction],
    )
    return model


def cross_modality_model_with_transformer(raw_eeg_shape, spectrogram_shape, mri_shape):
    
    # Encoders
    raw_eeg_enc = raw_eeg_encoder(raw_eeg_shape)
    spectrogram_enc = spectrogram_encoder(spectrogram_shape)
    eeg_combined_input = layers.concatenate([raw_eeg_enc.output, spectrogram_enc.output])
    eeg_shared_latent = layers.Dense(128, activation='relu')(eeg_combined_input)

    mri_enc = mri_encoder(mri_shape)
    mri_latent = mri_enc.output

    # Combine the latents from EEG and MRI
    combined_latent = layers.concatenate([eeg_shared_latent, mri_latent])
    combined_latent = layers.Dense(128, activation='relu')(combined_latent)

    # Decoder models for EEG and MRI
    eeg_decoder_model = eeg_decoder(latent_dim=128, output_shape=spectrogram_shape)
    mri_decoder_model = mri_decoder(latent_dim=128, output_shape=mri_shape)

    eeg_reconstruction = eeg_decoder_model(combined_latent)
    mri_reconstruction = mri_decoder_model(combined_latent)

    # Hopfield-like retrieval
    hopfield_mapping = layers.Dense(128, activation='relu')(combined_latent)
    hopfield_mri_reconstruction = mri_decoder_model(hopfield_mapping)

    # Final model
    model = models.Model(
        inputs=[raw_eeg_enc.input, spectrogram_enc.input, mri_enc.input],
        outputs=[eeg_reconstruction, mri_reconstruction, hopfield_mri_reconstruction],
    )
    return model

# Define shapes for the inputs
raw_eeg_shape = (128, 1)
spectrogram_shape = (128, 132, 1)
mri_shape = (76, 76, 40, 1)

# Initialize the model
model = cross_modality_model_with_transformer(raw_eeg_shape, spectrogram_shape, mri_shape)

# Compile the model
model.compile(optimizer='adam', loss=['mse', 'mse', 'mse'])

# Show the model summary
model.summary()


