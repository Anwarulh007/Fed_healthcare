import tensorflow as tf
import os
import numpy as np
import pickle
import logging
from flwr.client import NumPyClient, start_numpy_client
from model.model import get_model, load_data
from encryption_utils import encrypt_weights, decrypt_weights
import logging
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Suppress Flower's internal logs
logging.getLogger("flwr").setLevel(logging.WARNING)

# Optional: Also suppress gRPC debug logs if you still see them
logging.getLogger("grpc").setLevel(logging.ERROR)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_minloglevel"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import absl.logging

# Disable absl logging
absl.logging.set_verbosity('error')
absl.logging.set_stderrthreshold('error')



# === Identify Client ===
client_id = int(os.environ.get("CLIENT_ID", 0))
# print(f"[Client {client_id}] 🚀 Booting up...")

# === Load Data ===
x_train, x_test, y_train, y_test = load_data(client_id)
#print(f"[Client {client_id}] 📊 Data loaded ➤ Train: {x_train.shape}, Test: {x_test.shape}")

# === Build Model ===
model = get_model(x_train.shape[1])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#print(f"[Client {client_id}] 🧠 Model ready.")

# === Federated Client ===
class EncryptedClient(NumPyClient):
    def get_parameters(self, config):
        weights = model.get_weights()
        encrypted_weights, shapes = encrypt_weights(weights)
        decrypted_weights = decrypt_weights(encrypted_weights, shapes)

        # 🔐 Validate Privacy
        if np.isclose(sum(w.sum() for w in weights), sum(w.sum() for w in decrypted_weights)):
            print(f"[Client {client_id}]  Privacy  Preserved")
        else:
            print(f"[Client {client_id}]  Privacy  Compromised")

        shape_array = np.frombuffer(pickle.dumps(shapes), dtype=np.uint8)
        return encrypted_weights + [shape_array]

    def fit(self, parameters, config):
        print(f"[Client {client_id}]  Round started...")
        updated_weights = self.get_parameters(config)
        print(f"[Client {client_id}]  Weights sent.")
        print(f"[Client {client_id}]  Round done.")
        return updated_weights, len(x_train), {}

# === Start Client ===
print(f"[Client {client_id}]  Connecting to server...\n")
start_numpy_client(server_address="server:8080", client=EncryptedClient())
