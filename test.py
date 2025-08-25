import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_sampling_log(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    entries = content.split('--- Start ---')
    data = []

    for entry in entries:
        if not entry.strip():
            continue

        # Extract property and number of atoms
        property_match = re.search(r'Property:\s*(\w+)', entry)
        atoms_match = re.search(r'Sampling molecules with (\d+) atoms', entry)
        stats_match = re.search(
            r'mean_of_means: ([\d\.\-e]+)\s+'
            r'std_of_means_of_means: ([\d\.\-e]+)\s+'
            r'median_of_means: ([\d\.\-e]+)\s+'
            r'mean_of_medians: ([\d\.\-e]+)\s+'
            r'std_of_medians_of_means: ([\d\.\-e]+)\s+'
            r'median_of_medians: ([\d\.\-e]+)', entry)

        if not (property_match and atoms_match and stats_match):
            continue

        property_name = property_match.group(1)
        num_atoms = int(atoms_match.group(1))
        stats = [float(g) for g in stats_match.groups()]

        data.append({
            'property': property_name,
            'num_atoms': num_atoms,
            'mean_of_means': stats[0],
            'std_of_means_of_means': stats[1],
            'median_of_means': stats[2],
            'mean_of_medians': stats[3],
            'std_of_medians_of_means': stats[4],
            'median_of_medians': stats[5],
        })

    return pd.DataFrame(data)

def plot_stats(df):
    props = df['property'].unique()
    for prop in props:
        subset = df[df['property'] == prop].sort_values('num_atoms')
        
        plt.figure()
        plt.plot(subset['num_atoms'], subset['mean_of_means'], marker='o', label='Mean of Means')
        plt.plot(subset['num_atoms'], subset['median_of_means'], marker='s', label='Median of Means')
        plt.plot(subset['num_atoms'], subset['mean_of_medians'], marker='^', label='Mean of Medians')
        plt.title(f'Statistics vs Atom Count for Property: {prop}')
        plt.xlabel('Number of Atoms')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{prop}_stats_plot.png')

# MAIN EXECUTION
txt_file = 'sampling_log.txt'  # Change this to your file path
df = parse_sampling_log(txt_file)
df.to_csv('sampling_summary.csv', index=False)
plot_stats(df)
