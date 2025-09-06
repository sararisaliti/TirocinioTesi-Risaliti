import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from codecarbon import EmissionsTracker
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.callbacks import DeadlineStopper, CheckpointSaver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from compute_NH import compute_NH_all
from eusnesn import EuSN, ESN
from aeon.datasets import load_classification

#Crea la cartella se non esiste
def ensure_dir(path: str | None):
    if path:
        os.makedirs(path, exist_ok=True)

#Ritorna l'ultima "energy_consumed" (kWh) dal CSV di CodeCarbon
def get_energy_csv_last(tracker) -> float:
    csv_name = getattr(tracker, "_output_file", "emissions.csv")
    out_dir  = getattr(tracker, "_output_dir", ".")
    csv_path = os.path.join(out_dir if out_dir else ".", csv_name)
    df = pd.read_csv(csv_path)
    return float(df.iloc[-1]["energy_consumed"])

np.random.seed(42)
tf.random.set_seed(42)

def safe_stop(tracker):
    result = tracker.stop()
    return result if result is not None else 0.0

#Dataset
dataset_name = "CharacterTrajectories"

#Caricamento del training set fornito dal dataset AEON
x_raw, y_raw = load_classification(name=dataset_name, split="train")
_, y_raw = np.unique(y_raw, return_inverse=True) #Conversione label a interi

#Split per train/validation (80% train, 20% val)
x_train_raw, x_val_raw, y_train, y_val = train_test_split(
    x_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

def normalize_series_fit(x):  #standardizzazione per timestep (ogni colonna è un istante temporale)
    Ns, Nx, Tx = x.shape  #numero di campioni, numero feature per timestep, numero di timestep
    scaler = StandardScaler().fit(x.reshape(-1, Tx)) #matrice (Ns*Nx, Tx), StandardScaler calcola media e deviazione standard per ogni colonna (quindi per ogni timestep)
    x_norm = scaler.transform(x.reshape(-1, Tx)).reshape(Ns, Nx, Tx)  #applico scale e riporto alla forma originale (Ns, Nx, Tx)
    return x_norm, scaler #per riutilizzare lo scaler

def normalize_series_apply(x, scaler): #per usare lo stesso scaler già fittato sul train
    Ns, Nx, Tx = x.shape
    x_norm = scaler.transform(x.reshape(-1, Tx)).reshape(Ns, Nx, Tx)
    return x_norm

#Fit scaler solo sul train, poi applico al val
x_train, scaler_ms = normalize_series_fit(x_train_raw)
x_val   = normalize_series_apply(x_val_raw, scaler_ms)

#Transposizione per avere (samples, timesteps, features)
x_train = np.transpose(x_train, (0, 2, 1))   #(Ns, Tx, Nx)
x_val   = np.transpose(x_val,   (0, 2, 1))

#Dimensioni e classi
Tx = x_train.shape[1] #Numero di timestep
Nx = x_train.shape[2] #Numero di features
Ny = len(np.unique(y_train)) #Numero di classi

#Calcolo NH ottimale da compute_NH (vincolati dal costo del reservoir con 512 unità fisse)
reservoir_units = 512
NH_SRNN, _, NH_GRU, _ = compute_NH_all(reservoir_units, Nx, Ny, Tx)
print(f"NH_GRU calcolato: {NH_GRU}")
print(f"NH_SRNN calcolato: {NH_SRNN}")

#Spazi di ricerca
space_ft = [
    Categorical([1e-5, 1e-4, 1e-3, 1e-2], name="learning_rate"),
    Categorical([16, 32, 64, 128], name="batch_size")
]

space_esn = [
    Real(0.5, 1.5, name="spectral_radius"),
    Categorical([1e-4, 1e-3, 1e-2, 1e-1, 1.0], name="leaky"),
    Real(0.1, 1.5, name="input_scaling"),
    Real(0.1, 1.5, name="bias_scaling"),
]

space_eusn = [
    Real(1e-4, 1.0, prior="log-uniform", name="epsilon"),
    Real(1e-4, 1.0, prior="log-uniform", name="gamma"),
    Real(0.1, 1.5, name="recurrent_scaling"),
    Real(0.1, 1.5, name="input_scaling"),
    Real(0.1, 1.5, name="bias_scaling"),
]

#Funzione obiettivo GRU: riceve gli iperpaarametri, addestra GRU sul set di training, valuta la accuracy sul validation set e restituisce l'accuracy negativa (per minimizzazione)
def gru_objective(params):
    lr, bs = params #parametri da minimizzare: learning rate e batch size
    
    model = Sequential([
        tf.keras.Input(shape=(Tx, Nx)),  #(timesteps, features)
        GRU(NH_GRU),
        Dense(Ny, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    early = EarlyStopping(monitor="val_accuracy", mode='max', patience=50, restore_best_weights=True, verbose=0)
    
    #Training del modello con i parametri correnti
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=5000, batch_size=bs, callbacks=[early], verbose=0)
    #Valutazione del modello sui dati di validazione
    _, acc = model.evaluate(x_val, y_val, verbose=0)
    
    #Stampa parametri e accuracy
    print(f"[GRU] lr={lr:.1e}, batch={bs} => acc={acc:.4f}")
    return -acc

deadline_cb = DeadlineStopper(total_time=24 * 60 * 60)  # 24 ore
checkpoint_cb = CheckpointSaver("gru_checkpoint.pkl", compress=9)

#Tracker model selection GRU
ensure_dir("cc_logs/gru_ms")  #cartella per il CSV
tracker_gru = EmissionsTracker(project_name="GRU_model_selection", save_to_file=True, output_dir="cc_logs/gru_ms")
tracker_gru.start()
start_gru = time()

#Ottimizzazione Bayesiana
res_gru = gp_minimize(
    func=gru_objective,
    dimensions=space_ft,
    n_calls=200, #Numero di valutazioni
    callback=[deadline_cb, checkpoint_cb], #Stop dopo 24 ore e salvataggio checkpoint
    random_state=42,
    verbose=True
)

end_gru = time()
emission_gru = tracker_gru.stop()

best_gru = {
    "learning_rate": res_gru.x[0],
    "batch_size": res_gru.x[1],
    "val_acc": -res_gru.fun,
    "tempo": end_gru - start_gru,
    "kwh": get_energy_csv_last(tracker_gru), 
    "co2": emission_gru if emission_gru is not None else 0.0
}

#Funzione obiettivo SRNN
def srnn_objective(params):
    lr, bs = params

    model = Sequential([
        tf.keras.Input(shape=(Tx, Nx)), #(timesteps, features)
        SimpleRNN(NH_SRNN),
        Dense(Ny, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    early = EarlyStopping(monitor="val_accuracy", mode='max', patience=50, restore_best_weights=True, verbose=0)

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=5000, batch_size=bs, callbacks=[early], verbose=0)
    _, acc = model.evaluate(x_val, y_val, verbose=0)

    print(f"[SRNN] lr={lr:.1e}, batch={bs} => acc={acc:.4f}")
    return -acc

deadline_cb_srnn = DeadlineStopper(total_time=24 * 60 * 60)
checkpoint_cb_srnn = CheckpointSaver("srnn_checkpoint.pkl", compress=9)

#Tracker model selection SRNN
ensure_dir("cc_logs/srnn_ms")
tracker_srnn = EmissionsTracker(project_name="SRNN_model_selection", save_to_file=True, output_dir="cc_logs/srnn_ms")
tracker_srnn.start()
start_srnn = time()

res_srnn = gp_minimize(
    func=srnn_objective,
    dimensions=space_ft,
    n_calls=200,
    callback=[deadline_cb_srnn, checkpoint_cb_srnn],
    random_state=42,
    verbose=True
)

end_srnn = time()
emission_srnn = tracker_srnn.stop()

best_srnn = {
    "learning_rate": res_srnn.x[0],
    "batch_size": res_srnn.x[1],
    "val_acc": -res_srnn.fun,
    "tempo": end_srnn - start_srnn,
    "kwh": get_energy_csv_last(tracker_srnn),   
    "co2": emission_srnn if emission_srnn is not None else 0.0
}

#Funzione obiettivo ESN
def esn_objective(params):
    spectral_radius, leaky, input_scaling, bias_scaling = params

    model = ESN(
        units=512,
        spectral_radius=spectral_radius,
        leaky=leaky,
        input_scaling=input_scaling,
        bias_scaling=bias_scaling
    )

    model.fit(x_train, y_train)
    acc = model.evaluate(x_val, y_val)

    print(f"[ESN] radius={spectral_radius:.3f}, leaky={leaky:.5f}, input={input_scaling:.3f}, bias={bias_scaling:.3f} => acc={acc:.4f}")
    return -acc

deadline_cb_esn = DeadlineStopper(total_time=24 * 60 * 60)
checkpoint_cb_esn = CheckpointSaver("esn_checkpoint.pkl", compress=9)

#Tracker model selection ESN
ensure_dir("cc_logs/esn_ms")
tracker_esn = EmissionsTracker(project_name="ESN_model_selection", save_to_file=True, output_dir="cc_logs/esn_ms")   
tracker_esn.start()
start_esn = time()

res_esn = gp_minimize(
    func=esn_objective,
    dimensions=space_esn,
    n_calls=200,
    callback=[deadline_cb_esn, checkpoint_cb_esn],
    random_state=42,
    verbose=True
)

end_esn = time()
emission_esn = tracker_esn.stop()

best_esn = {
    "spectral_radius": res_esn.x[0],
    "leaky": res_esn.x[1],
    "input_scaling": res_esn.x[2],
    "bias_scaling": res_esn.x[3],
    "val_acc": -res_esn.fun,
    "tempo": end_esn - start_esn,
    "kwh": get_energy_csv_last(tracker_esn),   
    "co2": emission_esn if emission_esn is not None else 0.0
}

#Funzione obiettivo EuSN
def eusn_objective(params):
    epsilon, gamma, recurrent_scaling, input_scaling, bias_scaling = params

    model = EuSN(
        units=512,
        epsilon=epsilon,
        gamma=gamma,
        recurrent_scaling=recurrent_scaling,
        input_scaling=input_scaling,
        bias_scaling=bias_scaling
    )

    model.fit(x_train, y_train)
    acc = model.evaluate(x_val, y_val)

    print(f"[EuSN] eps={epsilon:.5f}, gamma={gamma:.5f}, rec={recurrent_scaling:.3f}, input={input_scaling:.3f}, bias={bias_scaling:.3f} => acc={acc:.4f}")
    return -acc

deadline_cb_eusn = DeadlineStopper(total_time=24 * 60 * 60)
checkpoint_cb_eusn = CheckpointSaver("eusn_checkpoint.pkl", compress=9)

#Tracker model selection EuSN
ensure_dir("cc_logs/eusn_ms")
tracker_eusn = EmissionsTracker(project_name="EuSN_model_selection", save_to_file=True, output_dir="cc_logs/eusn_ms")   
tracker_eusn.start()
start_eusn = time()

res_eusn = gp_minimize(
    func=eusn_objective,
    dimensions=space_eusn,
    n_calls=200,
    callback=[deadline_cb_eusn, checkpoint_cb_eusn],
    random_state=42,
    verbose=True
)

end_eusn = time()
emission_eusn = tracker_eusn.stop()

best_eusn = {
    "epsilon": res_eusn.x[0],
    "gamma": res_eusn.x[1],
    "recurrent_scaling": res_eusn.x[2],
    "input_scaling": res_eusn.x[3],
    "bias_scaling": res_eusn.x[4],
    "val_acc": -res_eusn.fun,
    "tempo": end_eusn - start_eusn,
    "kwh": get_energy_csv_last(tracker_eusn),   
    "co2": emission_eusn if emission_eusn is not None else 0.0
}

#Riepilogo dei risultati della model selection
print("\nRIEPILOGO MODEL SELECTION\n")

print("[GRU] Iperparametri ottimali:")
print(f"  - Learning rate        : {best_gru['learning_rate']:.1e}")
print(f"  - Batch size           : {best_gru['batch_size']}")
print(f"  - Accuracy val         : {best_gru['val_acc']:.4f}")
print(f"  - Tempo impiegato      : {best_gru['tempo']:.2f} s")
print(f"  - Energia consumata    : {best_gru['kwh']:.4f} kWh")
print(f"  - Emissioni CO2        : {best_gru['co2']:.6f} kg\n")

print("[SRNN] Iperparametri ottimali:")
print(f"  - Learning rate        : {best_srnn['learning_rate']:.1e}")
print(f"  - Batch size           : {best_srnn['batch_size']}")
print(f"  - Accuracy val         : {best_srnn['val_acc']:.4f}")
print(f"  - Tempo impiegato      : {best_srnn['tempo']:.2f} s")
print(f"  - Energia consumata    : {best_srnn['kwh']:.4f} kWh")
print(f"  - Emissioni CO2        : {best_srnn['co2']:.6f} kg\n")

print("[ESN] Iperparametri ottimali:")
print(f"  - Spectral radius      : {best_esn['spectral_radius']:.3f}")
print(f"  - Leaky integrator     : {best_esn['leaky']:.5f}")
print(f"  - Input scaling        : {best_esn['input_scaling']:.3f}")
print(f"  - Bias scaling         : {best_esn['bias_scaling']:.3f}")
print(f"  - Accuracy val         : {best_esn['val_acc']:.4f}")
print(f"  - Tempo impiegato      : {best_esn['tempo']:.2f} s")
print(f"  - Energia consumata    : {best_esn['kwh']:.4f} kWh")
print(f"  - Emissioni CO2        : {best_esn['co2']:.6f} kg\n")

print("[EuSN] Iperparametri ottimali:")
print(f"  - Epsilon              : {best_eusn['epsilon']:.5f}")
print(f"  - Gamma                : {best_eusn['gamma']:.5f}")
print(f"  - Recurrent scaling    : {best_eusn['recurrent_scaling']:.3f}")
print(f"  - Input scaling        : {best_eusn['input_scaling']:.3f}")
print(f"  - Bias scaling         : {best_eusn['bias_scaling']:.3f}")
print(f"  - Accuracy val         : {best_eusn['val_acc']:.4f}")
print(f"  - Tempo impiegato      : {best_eusn['tempo']:.2f} s")
print(f"  - Energia consumata    : {best_eusn['kwh']:.4f} kWh")
print(f"  - Emissioni CO2        : {best_eusn['co2']:.6f} kg\n")

#Caricamento test set
x_test, y_test = load_classification(name=dataset_name, split="test")
_, y_test = np.unique(y_test, return_inverse=True)

#Funzione per GRU e SRNN (ft), 3 run
def run_ft_model(model_name, model_builder, x_train, y_train, x_test, y_test, n_runs=3):
    results = []

    for i in range(n_runs):
        print(f"[{model_name}] Run {i+1}/3")

        #TRAINING
        out_tr = f"cc_logs/{model_name}_train_run{i+1}"
        ensure_dir(out_tr)
        tracker_tr = EmissionsTracker(project_name=f"{model_name}_Train_Run{i+1}", save_to_file=True, output_dir=out_tr)   
        tracker_tr.start()
        t0 = time()
        model = model_builder()
        batch_size = model.bs
        early = EarlyStopping(monitor="val_accuracy", mode='max', patience=50, restore_best_weights=True, verbose=0)
        history = model.fit(
            x_train, y_train,
            epochs=5000, batch_size=batch_size, validation_split=0.05,
            callbacks=[early], verbose=0
        )

        t1 = time()
        co2_train = safe_stop(tracker_tr)
        
        #Per stampare anche l'accuracy del train all'epoca con migliore val_accuracy (== pesi ripristinati), non inclusa nel time e co2
        best_epoch = int(np.argmax(history.history["val_accuracy"]))
        acc_tr = float(history.history["accuracy"][best_epoch])

        #TEST
        out_ts = f"cc_logs/{model_name}_test_run{i+1}"
        ensure_dir(out_ts)
        tracker_ts = EmissionsTracker(project_name=f"{model_name}_Test_Run{i+1}", save_to_file=True, output_dir=out_ts)   
        tracker_ts.start()
        t2 = time()
        y_pred = np.argmax(model.predict(x_test), axis=1)
        acc = accuracy_score(y_test, y_pred)
        t3 = time()
        co2_test = safe_stop(tracker_ts)

        results.append({
            "accuracy_tr": acc_tr,
            "accuracy": acc,
            "tempo_tr": t1 - t0,
            "tempo_ts": t3 - t2,
            "kwh_tr": get_energy_csv_last(tracker_tr),   
            "kwh_ts": get_energy_csv_last(tracker_ts),   
            "co2_tr": co2_train,
            "co2_ts": co2_test
        })

    return results

#GRU
def build_gru_model():
    model = Sequential([
        tf.keras.Input(shape=(Tx, Nx)),
        GRU(NH_GRU),
        Dense(Ny, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=best_gru["learning_rate"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.bs = best_gru["batch_size"] 
    return model

#SRNN
def build_srnn_model():
    model = Sequential([
        tf.keras.Input(shape=(Tx, Nx)),
        SimpleRNN(NH_SRNN),
        Dense(Ny, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=best_srnn["learning_rate"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.bs = best_srnn["batch_size"]  
    return model

#Funzione per ESN e EuSN (rc), 3 run
def run_rc_model(model_name, model_builder, x_train, y_train, x_test, y_test, n_runs=3):
    results = []

    for i in range(n_runs):
        print(f"[{model_name}] Run {i+1}/3")

        #TRAINING
        out_tr = f"cc_logs/{model_name}_train_run{i+1}"
        ensure_dir(out_tr)
        tracker_tr = EmissionsTracker(project_name=f"{model_name}_Train_Run{i+1}", save_to_file=True, output_dir=out_tr)   
        tracker_tr.start()
        t0 = time()
        model = model_builder()
        model.fit(x_train, y_train)
        t1 = time()
        co2_train = safe_stop(tracker_tr)
        acc_tr = model.evaluate(x_train, y_train) #per stampare anche accuracy del training set, non inclusa nel time e co2

        #TEST
        out_ts = f"cc_logs/{model_name}_test_run{i+1}"
        ensure_dir(out_ts)
        tracker_ts = EmissionsTracker(project_name=f"{model_name}_Test_Run{i+1}", save_to_file=True, output_dir=out_ts)   
        tracker_ts.start()
        t2 = time()
        acc = model.evaluate(x_test, y_test)
        t3 = time()
        co2_test = safe_stop(tracker_ts)

        results.append({
            "accuracy_tr": acc_tr,
            "accuracy": acc,
            "tempo_tr": t1 - t0,
            "tempo_ts": t3 - t2,
            "kwh_tr": get_energy_csv_last(tracker_tr),   
            "kwh_ts": get_energy_csv_last(tracker_ts),   
            "co2_tr": co2_train,
            "co2_ts": co2_test
        })

    return results

#ESN
def build_esn_model():
    return ESN(
        units=512,
        spectral_radius=best_esn["spectral_radius"],
        leaky=best_esn["leaky"],
        input_scaling=best_esn["input_scaling"],
        bias_scaling=best_esn["bias_scaling"]
    )

#EuSN
def build_eusn_model():
    return EuSN(
        units=512,
        epsilon=best_eusn["epsilon"],
        gamma=best_eusn["gamma"],
        recurrent_scaling=best_eusn["recurrent_scaling"],
        input_scaling=best_eusn["input_scaling"],
        bias_scaling=best_eusn["bias_scaling"]
    )

#Riaddestramento su tutto il training set (train + val), rifitto lo scaler, poi test su test set
y_train_full = np.concatenate([y_train, y_val])

#Fit nuovo scaler sul train_full e applico al test
x_train_full, scaler_full = normalize_series_fit(
    np.concatenate([x_train_raw, x_val_raw])
)
x_test = normalize_series_apply(x_test, scaler_full)

#Trasposizione nuova per Keras
x_train_full = np.transpose(x_train_full, (0, 2, 1))
x_test       = np.transpose(x_test,       (0, 2, 1))

gru_runs = run_ft_model("GRU", build_gru_model, x_train_full, y_train_full, x_test, y_test)
srnn_runs = run_ft_model("SRNN", build_srnn_model, x_train_full, y_train_full, x_test, y_test)
esn_runs = run_rc_model("ESN", build_esn_model, x_train_full, y_train_full, x_test, y_test)
eusn_runs = run_rc_model("EuSN", build_eusn_model, x_train_full, y_train_full, x_test, y_test)

#Stampa risultati di ogni run
for model_name, runs in zip(
    ["GRU", "SRNN", "ESN", "EuSN"],
    [gru_runs, srnn_runs, esn_runs, eusn_runs] #Acoppio il modello con i suoi risultati delle run
):
    print(f"\n[{model_name}] Risultati delle 3 run:")
    print("  - Acc. TR   :", [f"{r['accuracy_tr']:.6f}" for r in runs])
    print("  - Accuracy  :", [f"{r['accuracy']:.6f}" for r in runs])
    print("  - Tempo TR  :", [f"{r['tempo_tr']:.6f} s" for r in runs])
    print("  - Tempo TS  :", [f"{r['tempo_ts']:.6f} s" for r in runs])
    print("  - kWh TR    :", [f"{r['kwh_tr']:.6f}" for r in runs])
    print("  - kWh TS    :", [f"{r['kwh_ts']:.6f}" for r in runs])
    print("  - CO₂ TR    :", [f"{r['co2_tr']:.6f} kg" for r in runs])
    print("  - CO₂ TS    :", [f"{r['co2_ts']:.6f} kg" for r in runs])

#Calcolo media e std
def aggregate_results(name, modelsel_time, modelsel_kwh, modelsel_co2, run_results):
    def mean_std(values):
        return f"{np.mean(values):.6f} ± {np.std(values, ddof=1):.6f}" # ddof=1 deviazione standard campionaria

    return { #Le chiavi sono tuple, le voci “MS” sono singole (nessuna deviazione standard: un solo valore); TR/TS sono media ± std sulle 3 run
        ("", "Modello"): name,
        ("Tempo", "MS"): f"{modelsel_time:.2f}",
        ("Tempo", "TR"): mean_std([r["tempo_tr"] for r in run_results]),
        ("Tempo", "TS"): mean_std([r["tempo_ts"] for r in run_results]),
        ("kWh", "MS"): f"{modelsel_kwh:.4f}",
        ("kWh", "TR"): mean_std([r["kwh_tr"] for r in run_results]),
        ("kWh", "TS"): mean_std([r["kwh_ts"] for r in run_results]),
        ("CO2", "MS"): f"{modelsel_co2:.4f}",
        ("CO2", "TR"): mean_std([r["co2_tr"] for r in run_results]),
        ("CO2", "TS"): mean_std([r["co2_ts"] for r in run_results]),
        ("Accuracy", "TR"): mean_std([r["accuracy_tr"] for r in run_results]),
        ("Accuracy", "TS"): mean_std([r["accuracy"] for r in run_results])
    }

#Lista risultati per tutti i modelli
final_results = [
    aggregate_results("GRU", best_gru["tempo"], best_gru["kwh"], best_gru["co2"], gru_runs),
    aggregate_results("SRNN", best_srnn["tempo"], best_srnn["kwh"], best_srnn["co2"], srnn_runs),
    aggregate_results("ESN", best_esn["tempo"], best_esn["kwh"], best_esn["co2"], esn_runs),
    aggregate_results("EuSN", best_eusn["tempo"], best_eusn["kwh"], best_eusn["co2"], eusn_runs),
]

df_results_multi = pd.DataFrame(final_results)

column_order = [
    ("", "Modello"),
    ("Tempo", "MS"), ("Tempo", "TR"), ("Tempo", "TS"),
    ("kWh", "MS"), ("kWh", "TR"), ("kWh", "TS"),
    ("CO2", "MS"), ("CO2", "TR"), ("CO2", "TS"),
    ("Accuracy", "TR"),
    ("Accuracy", "TS")
]
df_results_multi = df_results_multi[column_order]
df_results_multi.columns = pd.MultiIndex.from_tuples(df_results_multi.columns) #Per  una tabella con intestazioni su due righe (livello1 sopra, livello2 sotto).

print(f"\nTabella finale: {dataset_name}\n")
print(df_results_multi.to_string(index=False))
