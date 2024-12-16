import os
import random as rn
import sys

import numpy as np
import biobox as bb
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# seeds for reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(3230)
rn.seed(3231)
tf.random.set_seed(1)

# File and model parameters
infile = sys.argv[1]
encoding_dim = 2  # Latent space dimension
BATCH_SIZE = 64
EPOCHS = 5
NUM_SAMPLES = 100000  # Number of new conformations to generate

def main():
    # Load data
    train_infile = infile.replace(".pdb", "_train.dat")
    test_infile = infile.replace(".pdb", "_test.dat")

    x_train_orig = np.loadtxt(train_infile)
    x_test_orig = np.loadtxt(test_infile)

    # Normalize input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train_orig)
    x_test = scaler.fit_transform(x_test_orig)

    # Build VAE model
    encoder, decoder, sampling_model = vae_structure(x_train)

    # Train VAE model
    history = sampling_model.fit(x_train, x_train,
                                 epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 validation_data=(x_test, x_test))

    # save reconstructed structures
    x_train = scaler.fit_transform(x_train_orig)
    rc_train = sampling_model.predict(x_train)
    reconstruct_train = scaler.inverse_transform(rc_train)
    decoded_reshaped_train = reconstruct_train.reshape(reconstruct_train.shape[0],
                                                       int(reconstruct_train.shape[1] / 3), 3)

    x_test = scaler.fit_transform(x_test_orig)
    rc_test = sampling_model.predict(x_test)
    reconstruct_test = scaler.inverse_transform(rc_test)
    decoded_reshaped_test = reconstruct_test.reshape(reconstruct_test.shape[0], int(reconstruct_test.shape[1] / 3), 3)

    # Calculate evaluation metrics
    spr = spearman_corr(x_train_orig, reconstruct_train)
    spr_test = spearman_corr(x_test_orig, reconstruct_test)

    print("> spearman correlation coefficients:\n  train set: %.3f\n  test set: %.3f" % (spr, spr_test))

    # Calculate mean squared error 
    mse_train = np.mean(np.square(x_train_orig - reconstruct_train))
    mse_test = np.mean(np.square(x_test_orig - reconstruct_test))
    print("> Mean squared error:\n  Train set: %.3f\n  Test set: %.3f" % (mse_train, mse_test))

    # Assuming x_train or x_test is already preprocessed and available
    plot_latent_vectors(encoder, x_train, save_file="latent_vectors.csv")

    # Load initial molecule data to get the number of atoms (assuming you want to use biobox)
    M = bb.Molecule()
    M.import_pdb(infile)
    #idx = M.atomselect("*", "*", ["CA", "C", "N", "O"], get_index=True)[1]
    idx = M.atomselect("*", "*", ["N", "CA", "CB", "CG", "SD", "CE", "C", "O", "CD", "OE1", "NE2", "ND1", "CE1", "CD2", "CD1", "CG1", "CG2", "CZ", "CE2", "OG", "NZ", "OE2", "OD1", "OD2", "OG1", "NE", "NH1", "NH2", "ND2", "OH", "OT1", "OT2"], get_index=True)[1]

    num_atoms = len(idx)  # Number of atoms in the selection
    all_new_conformations = []
    # Generate new protein conformations
    for i in range(NUM_SAMPLES):
        sampled_latent_point = np.random.normal(loc=0.0, scale=1.0, size=(1, encoding_dim))  # Sample from latent space
        generated_conformation = decoder.predict(sampled_latent_point)
        reconstructed = scaler.inverse_transform(generated_conformation)

        # Ensure correct reshaping: Make sure the generated conformation has shape (#atoms, 3)
        decoded_reshaped = reconstructed.reshape(num_atoms, 3)
        all_new_conformations.append(decoded_reshaped)

    all_new_conformations_np = np.array(all_new_conformations)
    M4 = M.get_subset(idxs=idx, conformations=[0])
    M4.coordinates = all_new_conformations_np  # Assign the generated coordinates to the biobox molecule
    M4.write_pdb("./data/%s" % infile.replace(".pdb", ".generated.pdb"))

    print(f"> {NUM_SAMPLES} new protein conformations have been generated and saved.")

def plot_latent_vectors(encoder, x_data, bins=50, save_file="latent_vectors.csv"):
    """
    Plots a heatmap showing the distribution of latent vectors in a 2D latent space
    and saves the latent vector values to a CSV file.
    
    Parameters:
    - encoder: Trained encoder model to encode data into the latent space.
    - x_data: Data to encode into the latent space (e.g., x_train or x_test).
    - bins: Number of bins to use for the 2D histogram (higher values give finer resolution).
    - save_file: File path to save the latent vector values as a CSV file.
    """
    # Encode the data to get the latent vectors (mean of distribution)
    latent_vectors = encoder.predict(x_data)
    
    # Extract the two latent dimensions
    z1 = latent_vectors[:, 0]
    z2 = latent_vectors[:, 1]
    
    # Save the latent vectors to a CSV file
    latent_df = pd.DataFrame({"Latent Dimension 1": z1, "Latent Dimension 2": z2})
    latent_df.to_csv(save_file, index=False)
    print(f"Latent vector values saved to {save_file}")

def spearman_corr(y_true, y_pred):
    """calculate mean Spearman correlation for each data sample."""
    sprm = 0
    for i in range(y_true.shape[0]):
        sprm += stats.spearmanr(y_true[i], y_pred[i])[0]
    sprm = sprm / y_true.shape[0]
    return sprm


def vae_structure(x_train):
    # Encoder
    input_prot = Input(shape=(x_train.shape[1],))

    encoded = Dense(1024, activation='relu')(input_prot)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    encode_mean = Dense(encoding_dim)(encoded)
    encode_log_var = Dense(encoding_dim)(encoded)

    # Lambda layer to sample from the latent space
    sample = Lambda(sampling, name='sampling')([encode_mean, encode_log_var])

    encoder = Model(input_prot, encode_mean)  # Create the encoder model

    # Decoder
    latent_input = Input(shape=(encoding_dim,))
    decoded = Dense(16, activation='relu')(latent_input)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(x_train.shape[1], activation='sigmoid')(decoded)

    decoder = Model(latent_input, decoded)  # Create the decoder model

    # Full VAE model
    sampling_model = Model(input_prot, decoder(sample))

    # Define loss, optimizer
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    xent_loss = K.sum(msle(input_prot, decoder(sample)))
    kl_loss = -5e-4 * K.mean(1 + encode_log_var - K.square(encode_mean) - K.exp(encode_log_var))
    vae_loss = K.mean(xent_loss + kl_loss)
    sampling_model.add_loss(vae_loss)

    # Use Adam optimizer
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sampling_model.compile(optimizer=adam, metrics=['mean_squared_error'])
    #sampling_model.compile(optimizer='rmsprop', metrics=['mean_squared_error'])

    return encoder, decoder, sampling_model

def sampling(arg):
    mean, logvar = arg
    epsilon = K.random_normal(shape=K.shape(mean), mean=0.0, stddev=1.0)  # Standard normal distribution
    return mean + K.exp(0.5 * logvar) * epsilon  # Reparameterization trick

if __name__ == "__main__":
    main()
