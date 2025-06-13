from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional
from client.encryption_utils import decrypt_weights
import numpy as np
import pickle

class EncryptedFedAvg(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict]:

        print(f"\n🔁 [Round {server_round}] Aggregating updates...")

        if not results:
            print(f"⚠️  No results from clients.")
            return None, {}

        decrypted_weights = []
        total_examples = 0

        for i, (_, fit_res) in enumerate(results):
            try:
                params = parameters_to_ndarrays(fit_res.parameters)

                if not (params and isinstance(params[-1], np.ndarray) and params[-1].dtype == np.uint8):
                    print(f"⚠️  Client {i+1} missing shape metadata.")
                    continue

                # Extract shapes and decrypt
                shapes = pickle.loads(params[-1].tobytes())
                encrypted = params[:-1]
                weights = decrypt_weights(encrypted, shapes)

                decrypted_weights.append((fit_res.num_examples, weights))
                total_examples += fit_res.num_examples

            except Exception as e:
                print(f"❌ Client {i+1} decryption failed: {e}")
                continue

        if not decrypted_weights:
            print("🚫 No valid client updates.")
            return None, {}

        # Aggregate
        num_layers = len(decrypted_weights[0][1])
        avg_weights = []
        for i in range(num_layers):
            layer_sum = sum(num * weights[i] for num, weights in decrypted_weights)
            avg_weights.append(layer_sum / total_examples)

        print(f"✅ Aggregation done with {len(decrypted_weights)} clients.\n")
        return ndarrays_to_parameters(avg_weights), {}
