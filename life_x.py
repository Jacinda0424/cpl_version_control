from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import pandas as pd

match_score = 1
mismatch_penalty = -1
gap_penalty = -2

# Function to perform global alignment
def global_alignment(sequence_1, sequence_2, match_score, mismatch_penalty, gap_penalty):
    alighments = pairwise2.align.globalxx(sequence_1, sequence_2, one_alignment_only=True)
    
    if alignments:
        best_alignment = alignments[0]
        print(format_alignment(*best_alignment))
        return best_alignment
    else:
        print("No alignment found.")
        return None

# Filter out lines stat with '>8AA'
with open("100_known_species_Seq_2023.txt", "r") as file:
    100_known_species = file.readlines()
    filtered_lines = filter(lambda x: ">8AA_" not in x, lines)
    print(filtered_lines)

# Write the filtered content back to the output file
filtered_file = open("filtered_file","w").writelines(filtered_lines)

#df = pd.read_csv("100_known_species_Seq_2023.txt", names=["Sequence"], header=None)
#dff = pd.read_csv("Life_X_Query_Seq_2023.txt",)

# Read the file
with open("Life_X_Query_Seq_2023.txt", "r") as file:
    life_x_sequence = file.readlines()

# Perform alignment for each known species sequence
resluts = []
for known_species_sequence in filtered_file:
    alignment_result = global_alignment(life_x_sequence, known_species_sequence, match_score, mismatch_penalty, gap_penalty)

    # Extract aligned sequence
    if alignment_result:
        results.append(alignment_result)
        #aligned_life_x, aligned_known_species, _, _ = result
        #print("Aligned Life X:", aligned_life_x)
        #print("Aligned Known Species:", aligned_known_species)
df_results = pd.DataFrame(results, columns)
