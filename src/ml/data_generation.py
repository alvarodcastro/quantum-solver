from crypto.bb84.bb84 import BB84
from quantum_solver.quantum_solver import QuantumSolver

if __name__ == "__main__":
    token = None
    bb84 = BB84(token)
    bb84.qexecute = QuantumSolver(token).get_qexecute()
    bb84.qexecute.current_backend = bb84.qexecute.backends[0]
    
    iterations = 25
    noise_rate = 0.0

    # Iterate message_bit_length from 20 to 300 stepping by 10
    for message_bit_length in range(20, 301, 10):
        # Interception density from 0.0 to 1.0 in steps of 0.05
        densities = [x / 20 for x in range(0, 21)]  # 0.00, 0.05, ..., 1.00
        for interception_density in densities:
            for _ in range(iterations):
                if interception_density == 0.0:
                    # Balance the number of runs with zero interception density
                    for _ in range(4):
                        bb84._BB84__run_simulation(message_bit_length, interception_density, noise_rate)
                bb84._BB84__run_simulation(message_bit_length, interception_density, noise_rate)