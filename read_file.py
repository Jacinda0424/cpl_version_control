import pandas as pd
from Bio import pairwise2

df_x   = pd.read_csv("filtered_Life_X_Query_Seq_2023.txt", sep=" ", header=None)
df_100 = pd.read_csv("filtered_100_known_species_Seq_2023.txt", sep=" ", header=None)

for i in df_100:
    alignment = pairwise2.align.globalxx(df_x, i)
    print(alignment)
    
