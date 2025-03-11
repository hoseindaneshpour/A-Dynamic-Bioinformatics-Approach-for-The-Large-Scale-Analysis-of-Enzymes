#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import requests
from Bio.PDB import PDBParser
import os
import logomaker as lm
import math


# Entropy

def calculate_entropy(alignment, valid_amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    if not alignment:
        raise ValueError("Empty alignment provided.")
    num_sequences = len(alignment)
    sequence_length = len(alignment[0])
    entropy_values = [0] * sequence_length
    amino_acid_indices = dict((aa, i) for i, aa in enumerate(valid_amino_acids))
    
    # Calculate the number of non-gap characters at each position
    non_gap_counts = [0] * sequence_length
    for i in range(sequence_length):
        for sequence in alignment:
            amino_acid = sequence[i]
            if amino_acid != "-":
                non_gap_counts[i] += 1
    
    for i in range(sequence_length):
        amino_acid_counts = [0] * len(amino_acid_indices)
        for sequence in alignment:
            amino_acid = sequence[i]
            if amino_acid in amino_acid_indices:
                amino_acid_counts[amino_acid_indices[amino_acid]] += 1
        amino_acid_probabilities = [count / non_gap_counts[i] for count in amino_acid_counts]
        for probability in amino_acid_probabilities:
            if probability > 0:
                entropy_values[i] += -probability * math.log(probability, math.e)
    return entropy_values

# Reading the alignment from the file
msafile = "./data/....aln"
alignment = AlignIO.read(msafile, "clustal")
# Calculating the entropy values
entropy = calculate_entropy(alignment)
# Printing the entropy values
print(entropy)



positions = list(range(len(entropy)))
# Plot the entropy values
plt.figure(figsize=(14, 4))
plt.plot(positions, entropy, linestyle='-', color='b')
plt.xlabel("Position in Alignment")
plt.ylabel("Entropy")
plt.title("Shannon Entropy in Multiple Sequence Alignment")
plt.grid(True)
plt.show()




# Weight

def calculate_weights(msafile):
    # Read the alignment from the file
    alignment = AlignIO.read(msafile, "clustal")
    # Initialize an empty list to store weights
    weights = []
    # Iterate over each column in the alignment
    for column in zip(*alignment):  # Transpose the alignment
        # Count the number of non-gap symbols in the column
        non_gap_count = sum(1 for symbol in column if symbol != "-")
        # Calculate the total number of symbols in the column
        total_symbols = len(column)
        # Calculate the weight (fraction of non-gap symbols)
        weight = non_gap_count / total_symbols
        # Append the weight to the list
        weights.append(weight)
    return weights
# Example usage
msafile = "./data/....aln"
column_weights = calculate_weights(msafile)





plt.figure(figsize=(14, 4))
plt.bar(range(1, len(column_weights) + 1), column_weights)
plt.xlabel("Column Number")
plt.ylabel("Weight")
plt.title("Weights for Each Column in MSA")
plt.grid(axis="y", linestyle="--", alpha=0.7)
# Show the plot
plt.tight_layout()
plt.show()









