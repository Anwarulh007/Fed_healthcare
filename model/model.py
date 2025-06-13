import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_model(input_dim, verbose=False):
    # print(f"[🧠] Building model for input_dim = {input_dim}")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
  
    return model

def load_data(client_id):
    import pandas as pd

    data_paths = {
        1: "/app/readmission_tokyo_encoded.csv",
        2: "/app/readmission_London_encoded.csv",
        3: "/app/Encoded_Johns_Hospital_Data.csv",
        4: "/app/Encoded_Prince_Hospital_Data.csv"
    }

    if client_id not in data_paths:
        raise ValueError(f" Invalid client_id {client_id}. Expected one of {list(data_paths.keys())}.")

    df = pd.read_csv(data_paths[client_id])
    X = df.drop(columns=["readmission"]).astype("float32")
    y = df["readmission"].astype("int32")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(f"[📦 Client {client_id}] Data shape: {X.shape}")
    return x_train, x_test, y_train, y_test

def get_checkpoint_callback(client_id):
    path = f"logs/model_checkpoints/client{client_id}"
    os.makedirs(path, exist_ok=True)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(path, "ckpt-epoch{epoch}.weights.h5"),
        save_weights_only=True,
        save_best_only=True
    )
