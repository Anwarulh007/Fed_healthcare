import numpy as np

def encrypt_weights(weights):
    shapes = [w.shape for w in weights]
    encrypted = [np.frombuffer(w.tobytes(), dtype=np.uint8) for w in weights]
    return encrypted, shapes

def decrypt_weights(encrypted, shapes):
    return [
        np.frombuffer(enc.tobytes(), dtype=np.float32).reshape(shape)
        for enc, shape in zip(encrypted, shapes)
    ]
