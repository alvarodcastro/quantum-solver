#!/usr/bin/env python3

# Author: Francesco Fiorini francesco.fiorini@phd.unipi.it
# Credits: Daniel Escanez-Exposito (https://github.com/jdanielescanez/quantum-solver)

from qiskit import QuantumCircuit
from crypto.bb84.sender import Sender
from crypto.bb84.receiver import Receiver
import binascii
import pandas as pd
import time

BB84_SIMULATOR = 'BB84 SIMULATOR'
DATA = {
  'Algorithm': ['BB84'],
  'Backend': ['-'],
  'Num_bits': ['-'],
  'Interception Density': ['-'],
  'Alice Values': ['-'],
  'Alice Axes': ['-'],
  'Eve Values': ['-'],
  'Eve Axes': ['-'],
  'Bob Values': ['-'],
  'Bob Axes': ['-'],
  'Alice Key': ['-'],
  'Bob Key': ['-'],
  'Shared Key': ['-'],
  'Alice OTP': ['-'],
  'Bob OTP': ['-'],
  'Result': ['-'],
  'Encryption iteration times (ms)': ['-'],
  'Encryption time (ms)': ['-'],
  'Interception iteration times (ms)': ['-'],
  'Interception time (ms)': ['-'],
  'Decryption iteration times (ms)': ['-'],
  'Decryption time (ms)': ['-'],
  'Private key generation time (ms)': ['-'],
  'Key checking time (ms)': ['-'],
  'Shared key demonstration time (ms)': ['-'],
  'Total time (ms)': ['-'],
  'Shared differences': ['-'],
  'Shared key length': ['-'],
  'Shared BER': ['-'],
  'Full key differences': ['-'],
  'Full key length': ['-'],
  'Full key BER': ['-'],
  'Raw_sifted_errors': ['-'],
  'Raw_sifted_length': ['-'],
  'Raw_sifted_QBER': ['-'],
  'P_sample': ['-'],
  'ech': ['-'],
  'Sifted length': ['-'],
  'Sifting efficiency': ['-'],
  'Interception realized count': ['-'],
  'Interception realized ratio': ['-'],
  'Per-basis sifted length (Z)': ['-'],
  'Per-basis sifted length (X)': ['-'],
  'Per-basis QBER (Z)': ['-'],
  'Per-basis QBER (X)': ['-'],
  'Attack label': ['-']
}


class BB84Algorithm:
  def _normalize_single_row(self, data_dict):
    return {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in data_dict.items()
    }
  ## Generate a key for Alice and Bob
  def __generate_key(self, backend, original_bits_size, verbose,ech):
    key_gen_time = time.time()
    # Encoder Alice
    alice = Sender('Alice', original_bits_size)
    alice.set_values()
    alice.set_axes()
    start_time = time.time()
    message, iteration_times = alice.encode_quantum_message()
    time_ms = (time.time() - start_time) * 1000
    DATA["Encryption iteration times (ms)"] = str(iteration_times)
    DATA["Encryption time (ms)"] = str(time_ms)

    # Interceptor Eve
    
    bob_axes = [] # Bob share his axes
    eve = Receiver('Eve', original_bits_size)
    eve.set_axes()
    start_time = time.time()
    message, iteration_times = eve.decode_quantum_messageEve(message, self.measure_density, backend)
    
    time_ms = (time.time() - start_time) * 1000
    DATA["Interception iteration times (ms)"] = str(iteration_times)
    DATA["Interception time (ms)"] = str(time_ms)

    # Decoder Bob
    bob = Receiver('Bob', original_bits_size)
    bob.set_axes()
    
    start_time = time.time()
    message, iteration_times = bob.decode_quantum_messageBob(message, 1, backend)
    time_ms = (time.time() - start_time) * 1000
    DATA["Decryption iteration times (ms)"] = str(iteration_times)
    DATA["Decryption time (ms)"] = str(time_ms)

    # Alice - Bob Remove Garbage
    alice_axes = alice.axes # Alice share her axes
    bob_axes = bob.axes # Bob share his axes
    # Delete the difference
    alice.remove_garbage(bob_axes)
    bob.remove_garbage(alice_axes)

    # After alice.remove_garbage(bob_axes) and bob.remove_garbage(alice_axes)
    alice_sifted = alice.key  
    bob_sifted   = bob.key

    # Compute raw sifted QBER
    raw_counter = sum(1 for i in range(len(alice_sifted)) if alice_sifted[i] != bob_sifted[i])
    DATA["Raw_sifted_errors"] = str(raw_counter)
    DATA["Raw_sifted_length"] = str(len(alice_sifted))
    DATA["Raw_sifted_QBER"]  = str(raw_counter/len(alice_sifted))

    # New: sifting efficiency
    sifted_len = len(alice_sifted)
    DATA["Sifted length"] = str(sifted_len)
    DATA["Sifting efficiency"] = str(sifted_len / self.original_bits_size if self.original_bits_size > 0 else 0.0)

    # New: realized interception stats (how many times Eve actually measured)
    eve_realized = sum(1 for v in eve.values if v != -1)
    DATA["Interception realized count"] = str(eve_realized)
    DATA["Interception realized ratio"] = str(eve_realized / self.original_bits_size if self.original_bits_size > 0 else 0.0)

    # New: per-basis QBER on sifted bits
    # Recover basis tags for sifted positions in the same order as key bits
    sifted_axes = [alice_axes[i] for i in range(len(alice_axes)) if alice_axes[i] == bob_axes[i]]
    # Split keys by basis
    z_pairs = [(a, b) for (a, b, ax) in zip(alice_sifted, bob_sifted, sifted_axes) if ax == 0]
    x_pairs = [(a, b) for (a, b, ax) in zip(alice_sifted, bob_sifted, sifted_axes) if ax == 1]
    z_len = len(z_pairs)
    x_len = len(x_pairs)
    z_err = sum(int(a != b) for (a, b) in z_pairs)
    x_err = sum(int(a != b) for (a, b) in x_pairs)
    DATA["Per-basis sifted length (Z)"] = str(z_len)
    DATA["Per-basis sifted length (X)"] = str(x_len)
    DATA["Per-basis QBER (Z)"] = str(z_err / z_len if z_len > 0 else 0.0)
    DATA["Per-basis QBER (X)"] = str(x_err / x_len if x_len > 0 else 0.0)

    key_gen_time_ms = (time.time() - key_gen_time) * 1000
    DATA["Private key generation time (ms)"] = str(key_gen_time_ms)

    key_check_time = time.time()
    # Bob share some values of the key to check
    SHARED_SIZE = round(0.5 * len(bob.key))
    shared_key = bob.key[:SHARED_SIZE]

    if verbose:
      alice.show_values()
      alice.show_axes()
      DATA['Alice Values'] = str(alice.show_values())
      DATA['Alice Axes'] = str(alice.show_axes())

      eve.show_values()
      eve.show_axes()
      DATA['Eve Values'] = str(eve.show_values())
      DATA['Eve Axes'] = str(eve.show_axes())

      bob.show_values()
      bob.show_axes()
      DATA['Bob Values'] = str(bob.show_values())
      DATA['Bob Axes'] = str(bob.show_axes())

      alice.show_key()
      bob.show_key()
      DATA['Alice Key'] = str(alice.show_key())
      DATA['Bob Key'] = str(bob.show_key())

      print('\nShared Bob Key:')
      print(shared_key)
      DATA['Shared Key'] = str(shared_key)

    alice_key = alice.show_key()
    compare = alice_key[:len(shared_key)]
    # BER calculation
    counter = 0
    for i in range(len(shared_key)):
      if(shared_key[i] != compare[i]):
        counter = counter + 1
    DATA["Shared differences"] = str(counter)
    DATA["Shared key length"] = str(len(shared_key))
    DATA["Shared BER"] = str(counter/len(shared_key))
    sampleQber=counter/len(shared_key)

    psample = (sampleQber - ech) / (0.25 - ech**2) if abs(0.25 - ech**2) > 1e-9 else 0.0
    psample = max(0.0, min(1.0, psample))  # clamp for numerical robustness
    DATA['P_sample'] = str(psample)
    DATA['ech'] = str(ech)
    print('\nSample p: '+str(psample))

    DATA['Attack label'] = str(int(self.measure_density > 0))

    alice_key = alice.show_key()
    bob_key = bob.show_key()
    counter = 0
    for i in range(len(alice_key)):
      if(alice_key[i] != bob_key[i]):
        counter = counter + 1
    DATA["Full key differences"] = str(counter)
    DATA["Full key length"] = str(len(alice_key))
    DATA["Full key BER"] = str(counter/len(alice_key))

    # Alice check the shared key
    if alice.check_key(shared_key):
      shared_size = len(shared_key)
      alice.confirm_key(shared_size)
      bob.confirm_key(shared_size)
      if verbose:
        print('\nFinal Keys')
        alice.show_key()
        bob.show_key()
        print('\nSecure Communication!')
    elif verbose:
      print('\nUnsecure Communication! Eve has been detected intercepting messages\n')
      DATA['Result'] = 'Unsecure (intercepted)'

    key_check_time_ms = (time.time() - key_check_time) * 1000
    DATA["Key checking time (ms)"] = str(key_check_time_ms)
    return alice, bob

  ## Run the implementation of BB84 protocol
  def run(self, message, backend, original_bits_size, measure_density, n_bits, verbose,ech):
    ## The original size of the message
    self.original_bits_size = original_bits_size
    ## The probability of an interception occurring
    self.measure_density = measure_density

    alice, bob = self.__generate_key(backend, original_bits_size, verbose,ech)
    shared_key_time = time.time()
    if not (alice.is_safe_key and bob.is_safe_key):
      if verbose:
        print('❌ Message not sent')
        DATA['Result'] = 'Message not sent'
        DATA['Alice OTP'] = '-'
        DATA['Bob OTP'] = '-'
        DATA["Shared key demonstration time (ms)"] = '-'
        df = pd.DataFrame([self._normalize_single_row(DATA)])
        filename = 'data.xlsx'
        df.to_excel(filename, index=False)
      return False

    alice.generate_otp(n_bits)
    bob.generate_otp(n_bits)

    encoded_message = alice.xor_otp_message(message)
    decoded_message = bob.xor_otp_message(encoded_message)

    if verbose:
      alice.show_otp()
      bob.show_otp()
      DATA['Alice OTP'] = str(alice.show_otp())
      DATA['Bob OTP'] = str(bob.show_otp())

      print('\nInitial Message:')
      print(message)

      print('Encoded Message:')
      print(encoded_message)

      print('💡 Decoded Message:')
      print(decoded_message)

      shared_key_time_ms = (time.time() - shared_key_time) * 1000
      DATA["Shared key demonstration time (ms)"] = str(shared_key_time_ms)

      if message == decoded_message:
        print('\n✅ The initial message and the decoded message are identical')
        DATA['Result'] = 'Secure'
      else:
        print('\n❌ The initial message and the decoded message are different')
        DATA['Result'] = 'Different messages'
      df = pd.DataFrame([self._normalize_single_row(DATA)])
      filename = 'data.xlsx'
      df.to_excel(filename, index=False)

    
    return True
