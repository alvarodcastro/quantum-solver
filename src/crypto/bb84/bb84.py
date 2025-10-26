#!/usr/bin/env python3

# Author: Francesco Fiorini francesco.fiorini@phd.unipi.it
# Credits: Daniel Escanez-Exposito (https://github.com/jdanielescanez/quantum-solver)

from quantum_solver.quantum_solver import QuantumSolver
from crypto.bb84.bb84_algorithm import BB84Algorithm
import time
import matplotlib.pyplot as plt
import numpy as np
from halo import Halo
from numpy.random import randint
from random import SystemRandom, randrange
import string
from alive_progress import alive_bar
import pandas as pd
from math import ceil

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

## Main class of BB84 Simulator

class BB84:
  ## Constructor
  def __init__(self, token):
    ## The implemented protocol
    self.bb84_algorithm = BB84Algorithm()
    ## The IBMQ Experience token
    self.token = token

  ## Print header, get an QExecute and run main menu
  def run(self):
    ## A QExecute instance to execute the simulation
    self.qexecute = QuantumSolver(self.token).get_qexecute()
    self.__main_menu()

  ## Loop to run the main menu
  def __main_menu(self):
    while True:
      try:
        ## If a current backend has been selected
        self.is_selected_backend = self.qexecute.current_backend != None

        self.__show_options()
        self.__select_option()
      except Exception as _:
        pass

  ## Main menu
  def __show_options(self):
    is_guest_mode = self.qexecute.is_guest_mode()
    guest_mode_string = ' (Guest Mode)' if is_guest_mode else ''
    len_guest_mode_string = len(guest_mode_string)
    print('\n' + BB84_SIMULATOR + guest_mode_string)
    print('=' * (len(BB84_SIMULATOR) + len_guest_mode_string) + '\n')
    print('[1] See available Backends')
    print('[2] Select Backend')
    if self.is_selected_backend:
      print('\tCurrent Backend: ' + str(self.qexecute.current_backend))
    if self.is_selected_backend:
      print('[3] Run Algorithm')
    if self.is_selected_backend:
      print('[4] Experimental mode')
    print('[0] Exit\n')

  ## Run BB84 simulation once
  def __run_simulation(self, message, density,ech):
    DATA['Num_bits'] = message
    DATA["Interception Density"] = str(density)
    backend = self.qexecute.current_backend
    DATA['Backend'] = str(backend)
    N_BITS = 6
    # bits_size = len(message) * 5 * N_BITS
    bits_size = message

    possible_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    message = ''.join(SystemRandom().choice(possible_chars) for _ in range(ceil(message)))
    execution_description = str(self.qexecute.current_backend)
    execution_description += ' with message "'
    execution_description += message + '" and density "' + str(density) + '"'
    halo_text = 'Running BB84 simulation in ' + execution_description
    halo = Halo(text=halo_text, spinner="dots")
    try:
      halo.start()
      start_time = time.time()
      self.bb84_algorithm.run(message, backend, bits_size, density, N_BITS, True,ech)
      time_ms = (time.time() - start_time) * 1000
      halo.succeed()
      print('  BB84 simulation run in', str(time_ms), 'ms')
      DATA["Total time (ms)"] = str(time_ms)
      
      df1 = pd.read_excel('data.xlsx')
      df2 = pd.DataFrame([self.bb84_algorithm._normalize_single_row(DATA)])
      df1.replace('-', pd.NA, inplace=True)
      df2.replace('-', pd.NA, inplace=True)

      merged_df = df1.combine_first(df2)
      concat_row = merged_df.iloc[0].to_frame().T

      final_df = pd.read_excel('final_data.xlsx')
      file_concat = pd.concat([final_df, concat_row], axis=0)
      file_concat.to_excel('final_data.xlsx', index=False)

    except Exception as exception:
      halo.fail()
      print('Exception:', exception)

  ## Run an experiment of BB84 simulation
  def __experimental_mode(self, step_msg=5, len_msg_limit=75, density_step=0.05, density_min=0, density_max=1, repetition_instance=30):
    STEP_MSG = step_msg
    DENSITY_MIN = density_min
    DENSITY_MAX = density_max
    DENSITY_RANGE = int((DENSITY_MAX - DENSITY_MIN) / density_step)
    backend = self.qexecute.current_backend
    possible_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    image = np.zeros((DENSITY_RANGE + 1, len_msg_limit // STEP_MSG))
    x = list(range(STEP_MSG, len_msg_limit + 1, STEP_MSG))
    y = list(np.arange(DENSITY_MIN, DENSITY_MAX + density_step, density_step))
    start_time = time.time()
    print('\nRunning BB84 Simulator Experiment (in ' + str(backend) + '):')

    try:
      with alive_bar(len(x) * len(y) * repetition_instance) as bar:
        for j, density in enumerate(y):
          for i, len_message in enumerate(x):
            for _ in range(repetition_instance):
              message = ''.join(SystemRandom().choice(possible_chars) for _ in range(len_message))
              bits_size = len(message) * 5
              # time_start
              flag = self.bb84_algorithm.run(message, backend, bits_size, density, 1, False)
              # time_end
              image[j][i] += 1 if flag else 0
              bar()
          
    except Exception as exception:
      print('Exception:', exception)

    time_m = (time.time() - start_time)
    print('\n[$] Experiment Finished in ' + str(time_m) + ' s!')
    print('\n💡 Output:\n\nx: ' + str(x) + '\n\ny: ' + str(y))
    print('\nImage:\n' + str(image) + '\n')
    plt.figure(num='BB84 Simulator - Experimental Mode [' + str(backend) + ']')
    plt.pcolormesh(x, y, image, cmap='inferno', shading='auto')
    plt.colorbar(label='Times the protocol is determined safe')
    plt.xlabel('Message Length (number of bits)')
    plt.ylabel('Interception Density')
    plt.show()

  ## Select the option for main menu
  def __select_option(self):
    option = int(input('[&] Select an option: '))
    if option == 0:
      print()
      exit(0)
    elif option == 1:
      self.qexecute.print_avaiable_backends()
    elif option == 2:
      self.qexecute.select_backend()
    elif option == 3 and self.is_selected_backend:
      message = int(input('[&] Key length (bits): '))
      density = float(input('[&] Interception Density (float between 0 and 1): '))
      numberofiterations=int(input('[&] Number of iterations: '))
      ech=float(input('[&] Expected System noise error rate: '))
      for _ in range(numberofiterations):
        self.__run_simulation(message,density,ech)
    elif option == 4 and self.is_selected_backend:
      step_msg = int(input('[&] Specify message steps (number of bits): '))
      len_msg_limit = int(input('[&] Specify maximum message length (number of bits): '))
      density_step = float(input('[&] Specify density step: '))
      density_min = float(input('[&] Specify density min: '))
      density_max = float(input('[&] Specify density max: '))
      repetition_instance = int(input('[&] Specify number of repetitions for each instance: '))

      if len_msg_limit <= 0:
        raise ValueError('Maximum message length must be positive (> 0)')
      elif repetition_instance <= 0:
        raise ValueError('Number of repetitions for each instance must be positive (> 0)')
      elif density_step < 0 or density_step > 1:
        raise ValueError('Density step must be between 0 and 1 (∈ [0, 1])')
      else:
        if 1.0 % density_step != 0:
          density_step = 1 / round(1 / density_step)
        self.__experimental_mode(step_msg, len_msg_limit, density_step, density_min, density_max, repetition_instance)
    else:
      print('[!] Invalid option, try again')
