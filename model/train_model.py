import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping




num_splits = 3
raw_eeg_split = np.array_split(raw_eeg_train[:960], num_splits)
spectrogram_split = np.array_split(spec_eeg_train, num_splits)
mri_split = np.array_split(mri_train[:960], num_splits)


def train_on_chunk(chunk_idx, model, raw_eeg_split, spectrogram_split, mri_split, batch_size=8, epochs=10):

    raw_eeg_chunk = raw_eeg_split[chunk_idx]
    spectrogram_chunk = spectrogram_split[chunk_idx]
    mri_chunk = mri_split[chunk_idx]


    checkpoint_callback = ModelCheckpoint(
        filepath=f"model_weights_chunk_{chunk_idx}.keras",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )


    history = model.fit(
        [raw_eeg_chunk, spectrogram_chunk, mri_chunk],
        [spectrogram_chunk, mri_chunk,mri_chunk],  # Outputs
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping_callback,mri_visualizer]
    )
    return history


raw_eeg_shape = (128, 1)
spectrogram_shape = (128, 132, 1)
mri_shape = (76, 76, 40, 1)
model = cross_modality_model_with_transformer(raw_eeg_shape, spectrogram_shape, mri_shape)
model.compile(optimizer='adam', loss=['mse', 'mse', 'mse'])





chunk_idx = 0
for chunk_idx in range(num_splits):
  train_on_chunk(chunk_idx, model, raw_eeg_split, spectrogram_split, mri_split)

