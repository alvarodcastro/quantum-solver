#!/usr/bin/env python3

# Author: Francesco Fiorini francesco.fiorini@phd.unipi.it
# Credits: Daniel Escanez-Exposito (https://github.com/jdanielescanez/quantum-solver)

from crypto.bb84.participant import Participant
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers.aer import *
from qiskit.providers.fake_provider import *
from numpy.random import rand
import numpy as np
import time

pi = np.pi
simulator1 = Aer.get_backend('qasm_simulator')
## The Receiver entity in the BB84 implementation
class Receiver(Participant):
  ## Constructor
  def __init__(self, name='', original_bits_size=0):
    super().__init__(name, original_bits_size)

  ## Decode the message measuring the circuit (density-dependent)
  def decode_quantum_messageEve(self, message, density, backend):
    ## The values of the participant
    self.values = []
    iteration_times = []
    for i, qc in enumerate(message):
      start_time = time.time()
      qc.barrier()
      if rand() < density:
        if self.axes[i] == 1:
          qc.h(0)
        qc.measure(0, 0) 
        transpile(qc, backend)
        result = backend.run(qc, shots=1, memory=True).result()
        measured_bit = int(result.get_memory()[0])
        self.values.append(measured_bit)
        transpile(qc,simulator1)
        resultTrue=int(simulator1.run(qc, shots=1, memory=True).result().get_memory()[0]) 
        if resultTrue != measured_bit: 
          qc.x(0)
        if self.axes[i] == 1:
          qc.h(0)
      else:
        self.values.append(-1)
      time_ms = (time.time() - start_time) * 1000
      iteration_times.append(time_ms)
      
    return message, iteration_times
  
  def decode_quantum_messageBob(self, message, density, backend):
    ## The values of the participant
    self.values = []
    iteration_times = []
    for i, qc in enumerate(message):
      
      
      start_time = time.time()
      qc.barrier()
      if rand() < density:
        if self.axes[i] == 1:
          qc.h(0)
        qc.measure(0, 0)
        transpile(qc, backend)
        result = backend.run(qc, shots=1, memory=True).result()
        measured_bit = int(result.get_memory()[0])
        self.values.append(measured_bit)
      else:
        self.values.append(-1)
      time_ms = (time.time() - start_time) * 1000
      iteration_times.append(time_ms)
    return message, iteration_times

  
