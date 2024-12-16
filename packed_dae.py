import os
import random as rn
import sys
import numpy as np
import biobox as bb
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# seeds for reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(7023)
rn.seed(7033)
tf.random.set_seed(1)

# File and model parameters
infile = sys.argv[1]
encoding_dim = 2  # Latent space dimension
BATCH_SIZE = 64
EPOCHS = 5
NUM_SAMPLES = 100000  # Number of new conformations to generate
noise_factor = 0.5  # amount of noise to add to the input

def main():
    # Load data
    train_infile = infile.replace(".pdb", "_train.dat")
    test_infile = infile.replace(".pdb", "_test.dat")

    x_train_orig = np.loadtxt(train_infile)
    x_test_orig = np.loadtxt(test_infile)

    # Normalize input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train_orig)
    x_test = scaler.transform(x_test_orig)

    # Build Denoising Autoencoder (DAE) model
    encoder, decoder, dae_model = dae_structure(x_train.shape[1])

    # Generate noisy data
    x_train_noisy = add_noise(x_train, noise_factor)
    x_test_noisy = add_noise(x_test, noise_factor)

    # Train DAE model
    history = dae_model.fit(x_train_noisy, x_train,  # train on noisy inputs, target is clean input
                                 epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 validation_data=(x_test_noisy, x_test))

    # Reconstruct and inverse transform for evaluation



    rc_train = dae_model.predict(x_train_noisy)
    reconstruct_train = scaler.inverse_transform(rc_train)
    decoded_reshaped_train = reconstruct_train.reshape(reconstruct_train.shape[0],
                                                       int(reconstruct_train.shape[1] / 3), 3)


    rc_test = dae_model.predict(x_test_noisy)
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
    plot_latent_vectors(encoder, x_train, save_file="latent_vectors_dae.csv")

    # Generate new protein conformations
    generate_conformations(decoder, scaler, encoding_dim, NUM_SAMPLES, num_atoms=len(decoded_reshaped_train[0]), infile=infile)

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

    return sprm / y_true.shape[0]

def dae_structure(input_dim):
    """Build the Denoising Autoencoder structure."""
    # Encoder with noise layer
    input_img = Input(shape=(input_dim,))
    noisy_input = GaussianNoise(noise_factor)(input_img)  # Add Gaussian noise to input
    encoded = Dense(1024, activation='relu')(noisy_input)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    latent = Dense(encoding_dim, activation='linear')(encoded)  # Latent layer

    # Decoder layers
    latent_input = Input(shape=(encoding_dim,))
    decoded = Dense(16, activation='relu')(latent_input)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Build encoder and decoder models
    encoder = Model(inputs=input_img, outputs=latent)
    decoder = Model(inputs=latent_input, outputs=decoded)

    # Full DAE model (encoder + decoder)
    dae_model = Model(inputs=input_img, outputs=decoder(encoder(input_img)))
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    dae_model.compile(optimizer=adam, loss='mse')  # MSE loss for reconstruction

    return encoder, decoder, dae_model


def add_noise(data, noise_factor):
    """Add Gaussian noise to data."""
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(noisy_data, 0., 1.)  # Ensure data remains in valid range after adding noise


def generate_conformations(decoder, scaler, encoding_dim, num_samples, num_atoms, infile):
    """Generate new conformations from the decoder."""
    M = bb.Molecule()
    M.import_pdb(infile)
    #idx = M.atomselect("*", "*", ["CA", "C", "N", "O"], get_index=True)[1]
    idx = M.atomselect("*", "*", ["N", "CA", "CB", "CG", "SD", "CE", "C", "O", "CD", "OE1", "NE2", "ND1", "CE1", "CD2", "CD1", "CG1", "CG2", "CZ", "CE2", "OG", "NZ", "OE2", "OD1", "OD2", "OG1", "NE", "NH1", "NH2", "ND2", "OH", "OT1", "OT2"], get_index=True)[1]

    all_new_conformations = []
    for _ in range(num_samples):
        sampled_latent_point = np.random.normal(loc=0.0, scale=1.0, size=(1, encoding_dim))  # Sample from latent space
        generated_conformation = decoder.predict(sampled_latent_point)
        reconstructed = scaler.inverse_transform(generated_conformation)
        decoded_reshaped = reconstructed.reshape(num_atoms, 3)  # Reshape for molecular structure
        all_new_conformations.append(decoded_reshaped)

    # Save conformations to a PDB file
    all_new_conformations_np = np.array(all_new_conformations)
    M4 = M.get_subset(idxs=idx, conformations=[0])
    M4.coordinates = all_new_conformations_np
    M4.write_pdb("./data/%s" % infile.replace(".pdb", ".generated-dae.pdb"))

    print(f"> {num_samples} new protein conformations have been generated and saved.")


if __name__ == "__main__":
    main()
