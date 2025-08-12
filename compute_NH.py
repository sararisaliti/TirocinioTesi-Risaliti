from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

#Calcolo il numero minimo di unità nascoste (NH) per SimpleRNN affinché il costo dei parametri sia almeno pari a quello del reservoir.
def compute_srnn_NH_min(reservoir_units: int, Nx: int, Ny: int):
    NH = 1
    cost_reservoir = reservoir_units * Ny + Ny
    while True:
        cost_trainable = NH**2 + NH * (Nx + Ny + 1) + Ny
        if cost_trainable >= cost_reservoir:
            return NH, cost_trainable
        NH += 1

#Calcolo il numero minimo di unità nascoste (NH) per GRU affinché il costo dei parametri sia almeno pari a quello del reservoir.
def compute_gru_NH_min(reservoir_units: int, Nx: int, Ny: int, Tx: int):
    NH = 1
    cost_reservoir = reservoir_units * Ny + Ny
    while True:
        model = Sequential([
            GRU(NH, input_shape=(Tx, Nx)),
            Dense(Ny, activation="softmax")
        ])
        cost_trainable = model.count_params()
        if cost_trainable >= cost_reservoir:
            return NH, cost_trainable
        NH += 1

#Calcolo NH per SimpleRNN e GRU e restituisce tutti i risultati.
def compute_NH_all(reservoir_units: int, Nx: int, Ny: int, Tx: int):
    NH_SRNN, cost_srnn = compute_srnn_NH_min(reservoir_units, Nx, Ny)
    NH_GRU,  cost_gru  = compute_gru_NH_min(reservoir_units, Nx, Ny, Tx)
    return NH_SRNN, cost_srnn, NH_GRU, cost_gru

if __name__ == "__main__":
    from aeon.datasets import load_classification
    import numpy as np

    #Dataset 
    dataset_name = "Adiac"
    x_train_aeon, y_train_aeon = load_classification(name=dataset_name, split="train")
    x_train = np.transpose(x_train_aeon, (0, 2, 1))
    _, y_train = np.unique(y_train_aeon, return_inverse=True)

    Nx = x_train.shape[2]
    Tx = x_train.shape[1]
    Ny = len(np.unique(y_train))
    reservoir_units = 512

    NH_SRNN, cost_srnn, NH_GRU, cost_gru = compute_NH_all(reservoir_units, Nx, Ny, Tx)
    cost_reservoir = reservoir_units * Ny + Ny
    
    print(f"Dataset: {dataset_name}")
    print(f"Reservoir cost (EuSN/ESN): {cost_reservoir}")
    print(f"Nx: {Nx}, Tx: {Tx}, Ny: {Ny}, reservoir_units: {reservoir_units}")
    print(f"NH SimpleRNN: {NH_SRNN} (cost: {cost_srnn})")
    print(f"NH GRU:       {NH_GRU}  (cost: {cost_gru})")
