import flwr as fl
import sys
import os
import logging
import sys
sys.stdout.reconfigure(encoding='utf-8')

# === Suppress Logs ===
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_minloglevel"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import logging

# Suppress Flower's internal logs
logging.getLogger("flwr").setLevel(logging.WARNING)

# Optional: Also suppress gRPC debug logs if you still see them
logging.getLogger("grpc").setLevel(logging.ERROR)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import absl.logging

# Disable absl logging
absl.logging.set_verbosity('error')
absl.logging.set_stderrthreshold('error')




# === Import Custom Strategy ===
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'client'))
from strategy_secure import EncryptedFedAvg

# === Main Server Function ===
def main():
    print("\n [Server] Federated Learning Server Started")
    print("\n [Strategy] EncryptedFedAvg | Clients: 4 | Rounds: 5\n")

    strategy = EncryptedFedAvg(
        min_fit_clients=4,
        min_available_clients=4,
        min_evaluate_clients=4,
        fraction_fit=1.0,
        fraction_evaluate=1.0
    )

    print(" [Server] Listening at [::]:8080...\n")

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

    print("\n [Server] Training complete. Encrypted aggregation successful.")
    print(" [Server] Shutting down.\n")

if __name__ == "__main__":
    main()
