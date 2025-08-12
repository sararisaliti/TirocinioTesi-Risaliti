"""
Questo modulo implementa due modelli di Reservoir Computing (RC): EuSN (Euler State Network) ed ESN (Echo State Network).

Per entrambi i modelli:
- Il reservoir è implementato in TensorFlow/Keras come rete RNN personalizzata, utilizzando una cella custom (`EulerReservoirCell` per EuSN, `ReservoirCell` per ESN).
- La cella `EulerReservoirCell` integra la dinamica interna con uno schema di Eulero a passo fisso e dissipazione controllata, mentre `ReservoirCell` segue la dinamica standard delle Echo State Networks con controllo del raggio spettrale e integratore leaky.
- Il readout è implementato separatamente con `RidgeClassifierCV` di Scikit-learn, che esegue una classificazione lineare sugli stati finali del reservoir. Il parametro di regolarizzazione `alpha` viene selezionato automaticamente su un range logaritmico tra 1 e 10.000 (`np.logspace(0, 4, 20)`), con validazione incrociata a 4 fold.

I modelli allenano esclusivamente il readout: il reservoir resta fisso dopo l'inizializzazione casuale, in linea con il paradigma classico del Reservoir Computing.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import RidgeClassifierCV

#Definizione della cella EulerReservoir
class EulerReservoirCell(keras.layers.Layer):
    #Implementazione del livello di reservoir della Euler State Network
    #la funzione di transizione di stato è ottenuta tramite la discretizzazione di Eulero di un'ODE
    #la matrice dei pesi ricorrenti è vincolata ad avere una struttura anti-simmetrica (cioè, skewsymmetric)

    def __init__(self, units=512, input_scaling=1., bias_scaling=1.0, recurrent_scaling=1,
                 epsilon=0.01, gamma=0.001, activation=tf.nn.tanh, **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.recurrent_scaling = recurrent_scaling
        self.bias_scaling = bias_scaling
        self.epsilon = epsilon
        self.gamma = gamma
        self.activation = activation
        
        super().__init__(**kwargs)

    def build(self, input_shape):
        #Costruzione matrice dei pesi ricorrenti
        I = tf.linalg.eye(self.units)
        W = tf.random.uniform(shape=(self.units, self.units), minval=-self.recurrent_scaling, maxval=self.recurrent_scaling)
        self.recurrent_kernel = (W - tf.transpose(W) - self.gamma * I)
        
        #Costruzione matrice dei pesi in input
        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling, maxval=self.input_scaling)
        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = prev_output + self.epsilon * self.activation(input_part + self.bias + state_part)
        else:
            output = prev_output + self.epsilon * (input_part + self.bias + state_part)
        return output, [output]

#Definizione della cella Reservoir standard
class ReservoirCell(keras.layers.Layer):
    #Costruzione di un reservoir come strato dinamico nascosto per una RNN
    #utilizzo la legge circolare per inizializzare la matrice dei pesi ricorrenti, mantenendo controllato il raggio spettrale
    def __init__(self, units=512, input_scaling=1.0, bias_scaling=1.0, spectral_radius=0.99,
                 leaky=1, activation=tf.nn.tanh, **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        #Costruzione matrice dei pesi ricorrenti. Utilizzo la legge circolare per determinare i valori della matrice dei pesi ricorrenti
        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))
        W = tf.random.uniform(shape=(self.units, self.units), minval=-value, maxval=value)
        self.recurrent_kernel = W
        
        #Costruzione matrice dei pesi di input 
        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling, maxval=self.input_scaling)
        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = prev_output * (1 - self.leaky) + self.activation(input_part + self.bias + state_part) * self.leaky
        else:
            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky
        return output, [output]

#Modello EuSN estendendo keras.Model
class EuSN(keras.Model):
    #Implementazione modello di Euler State Network per la classificazione di serie temporali
    #L'architettura è composta da uno strato ricorrente con EulerReservoirCell, seguito da un readout denso addestrabile per la classificazione

    def __init__(self, units=512, input_scaling=1., bias_scaling=1.0, recurrent_scaling=1,
                 epsilon=0.01, gamma=0.001, readout_regularizer=1.0, activation=tf.nn.tanh, **kwargs):

        super().__init__(**kwargs)

        #Costruisco il reservoir come un modello Keras Sequential
        self.reservoir = keras.Sequential([
            keras.layers.RNN(cell=EulerReservoirCell(units=units,
                                                     input_scaling=input_scaling,
                                                     bias_scaling=bias_scaling,
                                                     recurrent_scaling=recurrent_scaling,
                                                     epsilon=epsilon,
                                                     gamma=gamma))
        ])
        alphas = np.logspace(-2, 2, 5) #5 valori logaritmicamente spaziati da 1e-2 a 1e2
        self.readout = RidgeClassifierCV(alphas=alphas) #Inizializzo il classificatore Ridge con selezione automatica del parametro di regolarizzazione alpha.
    def call(self, inputs):
        x_states = self.reservoir(inputs).numpy()
        return self.readout.predict(x_states)

    def fit(self, x, y, **kwargs):
        #Alleno solo il readout       
        x_states = self.reservoir(x).numpy()
        self.readout.fit(x_states, y)

    def evaluate(self, x, y):
        #Valuto solo il readout sui nuovi stati
        x_states = self.reservoir(x).numpy()
        return self.readout.score(x_states, y)
    
    
#Modello ESN estendendo keras.Model
class ESN(keras.Model):
    #Implementazione un modello di Echo State Network per la classificazione di serie temporali
    #L'architettura è composta da uno strato ricorrente con ReservoirCell, seguito da un readout denso addestrabile per la classificazione

    def __init__(self, units=512, input_scaling=1., bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1, readout_regularizer=1.0, activation=tf.nn.tanh, **kwargs):
            
        super().__init__(**kwargs)

        #Costruisco il reservoir
        self.reservoir = keras.Sequential([
            keras.layers.RNN(cell=ReservoirCell(units=units,
                                                 input_scaling=input_scaling,
                                                 bias_scaling=bias_scaling,
                                                 spectral_radius=spectral_radius,
                                                 leaky=leaky))
        ])
        alphas = np.logspace(-2, 2, 5) #5 valori logaritmicamente spaziati da 1e-2 a 1e2
        self.readout = RidgeClassifierCV(alphas=alphas) #Inizializzo il classificatore Ridge con selezione automatica del parametro di regolarizzazione alpha.
        
    def call(self, inputs):
        x_states = self.reservoir(inputs).numpy()
        return self.readout.predict(x_states)

    def fit(self, x, y, **kwargs):
        #Alleno solo il readout
        x_states = self.reservoir(x).numpy()  
        self.readout.fit(x_states, y)

    def evaluate(self, x, y):
        #Valuto solo il readout sui nuovi stati
        x_states = self.reservoir(x).numpy()
        return self.readout.score(x_states, y)
    
