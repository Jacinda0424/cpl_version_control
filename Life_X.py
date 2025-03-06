from Bio import pairwise2
from Bio.Seq import Seq
from Bio import AlignIO

seq1 = AlignIO.read(open("Life_X_Query_Seq_2023.txt"), header=None)
seq2 = AlignIO.parse(open("100_known_species_Seq_2023.txt"), ">8AA")
print(seq1)
for i in seq2:
    print(i)

