#!/usr/bin/env python3

# Author: Francesco Fiorini francesco.fiorini@phd.unipi.it
# Credits: Daniel Escanez-Exposito (https://github.com/jdanielescanez/quantum-solver)

from crypto.bb84.participant import Participant
from qiskit import QuantumCircuit
import numpy as np
import time

pi = np.pi

## The Sender entity in the BB84 implementation
class Sender(Participant):
  ## Constructor
  def __init__(self, name='', original_bits_size=0):
    super().__init__(name, original_bits_size)
    
  ## Encode the message (values) using a quantum circuit
  def encode_quantum_message(self):
    encoded_message = []
    iteration_times = []
    for i in range(len(self.axes)):
      start_time = time.time()
      # check what operation its using (x or h)
      # time_start
      qc = QuantumCircuit(1, 1) # quantum circuit 1x1: 1 qubit e 1 bit
      if self.values[i] == 1:
        qc.x(0)
      if self.axes[i] == 1:
        qc.h(0)
      encoded_message.append(qc)
      time_ms = (time.time() - start_time) * 1000
      iteration_times.append(time_ms)
    return encoded_message, iteration_times