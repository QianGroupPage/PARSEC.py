import re

def extract_data(filename):
    with open(filename, 'r') as file:
        content = file.readlines()

    # Initialize counters and accumulators
    scf_iter_count = 0
    total_diagonalization_time = 0.0
    total_hartree_potential_time = 0.0

    # Regular expressions for matching patterns
    iter_pattern = re.compile(r'SCF iter # (\d+)')
    diag_time_pattern = re.compile(r'Diagonalization time \[sec\] :\s+([\d\.]+)')
    hartree_time_pattern = re.compile(r'Hartree potential time \[sec\]:\s+([\d\.]+)')

    for line in content:
        # Count iterations by identifying each "SCF iter #" line
        iter_match = iter_pattern.search(line)
        if iter_match:
            scf_iter_count += 1

        # Accumulate diagonalization time
        diag_match = diag_time_pattern.search(line)
        if diag_match:
            total_diagonalization_time += float(diag_match.group(1))

        # Accumulate Hartree potential time
        hartree_match = hartree_time_pattern.search(line)
        if hartree_match:
            total_hartree_potential_time += float(hartree_match.group(1))

    # Calculate averages
    avg_diagonalization_time = total_diagonalization_time / scf_iter_count if scf_iter_count else 0
    avg_hartree_potential_time = total_hartree_potential_time / scf_iter_count if scf_iter_count else 0

    # Output the results
    results = {
        "Number of SCF iterations": scf_iter_count,
        "Average Diagonalization time [sec]": avg_diagonalization_time,
        "Average Hartree potential time [sec]": avg_hartree_potential_time
    }
    return results

# Example usage:
filename = "rsdft_parameter.out"
data = extract_data(filename)
print(data)
