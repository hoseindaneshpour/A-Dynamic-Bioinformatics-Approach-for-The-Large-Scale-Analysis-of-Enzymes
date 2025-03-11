#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ## Entropy

# In[2]:


# In this revision, I've added a check amino_acid in amino_acid_indices before incrementing the count of non-gap characters. 
#This ensures that only valid amino acids are considered for the count of non-gap characters.
#Additionally, I've added a condition if non_gap_counts[i] != 0 while calculating amino acid probabilities to avoid division by zero errors...
#if there are positions with no valid non-gap characters.

def calculate_entropy(alignment, valid_amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    if not alignment:
        raise ValueError("Empty alignment provided.")
    num_sequences = len(alignment)
    sequence_length = len(alignment[0])
    entropy_values = [0] * sequence_length
    amino_acid_indices = {aa: i for i, aa in enumerate(valid_amino_acids)}
    
    # Calculate the number of non-gap characters at each position
    non_gap_counts = [0] * sequence_length
    for i in range(sequence_length):
        for sequence in alignment:
            amino_acid = sequence[i]
            if amino_acid != "-" and amino_acid in amino_acid_indices:
                non_gap_counts[i] += 1
    
    for i in range(sequence_length):
        amino_acid_counts = [0] * len(amino_acid_indices)
        for sequence in alignment:
            amino_acid = sequence[i]
            if amino_acid in amino_acid_indices:
                amino_acid_counts[amino_acid_indices[amino_acid]] += 1
        amino_acid_probabilities = [count / non_gap_counts[i] for count in amino_acid_counts if non_gap_counts[i] != 0]
        for probability in amino_acid_probabilities:
            if probability > 0:
                entropy_values[i] += -probability * math.log(probability, math.e)
    return entropy_values

# Reading the alignment from the file
msafile = "./data/60seq.aln"
alignment = AlignIO.read(msafile, "clustal")
# Calculating the entropy values
entropy = calculate_entropy(alignment)
# Printing the entropy values
print(entropy)


# In[3]:


# the entropy values (stored in the 'entropy' list)
# Create x-axis (position) values
positions = list(range(len(entropy)))
# Plot the entropy values
plt.figure(figsize=(14, 4))
plt.plot(positions, entropy, linestyle='-', color='b')
plt.xlabel("Position in Alignment")
plt.ylabel("Entropy")
plt.title("Shannon Entropy in Multiple Sequence Alignment")
plt.grid(True)
plt.show()


# In[4]:


#correl between my entropy and weblogo entropy
logofile = "./data/logo60_noadj.txt"  # or with our prevoius logo60seq file
dfweights0 = pd.read_csv(logofile, sep="\t", header=7) 
entropy_values_weblogo = dfweights0['Entropy']
from scipy.stats import pearsonr
import scipy.stats as ss
correlation_coefficient, p_value = ss.pearsonr(entropy_values_weblogo, entropy)
# Print the correlation coefficient
print(f"Pearson correlation coefficient: {correlation_coefficient:.7f}")
print(f"P-value: {p_value:.5f}")


# In[5]:


def calculate_entropy(alignment):
    entropy_dict = {}
    for col_index in range(alignment.get_alignment_length()):
        col_data = alignment[:, col_index]
        # Count amino acid frequencies
        aa_counts = {}
        non_gap_count = 0  # Initialize non-gap count
        for aa in col_data:
            if aa != "-":  # Exclude gaps
                non_gap_count += 1
                if aa not in aa_counts:
                    aa_counts[aa] = 0
                aa_counts[aa] += 1
        # Calculate total number of sequences
        num_seqs = len(alignment)
        # Calculate entropy
        entropy = 0
        for aa, count in aa_counts.items():
            amino_acid_probability = count / non_gap_count  # Updated line
            if amino_acid_probability > 0:
                entropy += amino_acid_probability * math.log(amino_acid_probability)
        entropy *= -1  # Negate for convention as information content
        entropy_dict[col_index] = entropy
    return entropy_dict
# Read alignment file
alignment = AlignIO.read("data/60seq.aln", "clustal")
# Calculate entropy
entropy_dict = calculate_entropy(alignment)
# Print entropy values
for col_index, entropy in entropy_dict.items():
    print(f"Position {col_index+1}: {entropy:.4f}")


# In[6]:


# entropy values (stored in the 'entropy_dict' dictionary)
# Extract positions and entropy values
positions = list(entropy_dict.keys())
entropy_values = list(entropy_dict.values())
# Create the plot
plt.figure(figsize=(14, 4))
plt.plot(positions, entropy_values, linestyle='-', color='b')
plt.xlabel("Position in Alignment")
plt.ylabel("Entropy")
plt.title("Shannon Entropy in Multiple Sequence Alignment")
plt.grid(True)
plt.show()


# In[7]:


logofile = "./data/logo60_noadj.txt"  # or with our prevoius logo60seq file
dfweights0 = pd.read_csv(logofile, sep="\t", header=7) 
entropy_values_weblogo = dfweights0['Entropy']

from scipy.stats import pearsonr
import scipy.stats as ss
entropy_values_list = list(entropy_dict.values())
correlation_coefficient,p_value = pearsonr(entropy_values_weblogo, entropy_values_list)
# Print the correlation coefficient
print(f"Pearson correlation coefficient: {correlation_coefficient:.6f}")
print(f"P-value: {p_value:.5f}")


# previous method:

# In[8]:


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
msafile = "./data/60seq.aln"
alignment = AlignIO.read(msafile, "clustal")
# Calculating the entropy values
entropy = calculate_entropy(alignment)
# Printing the entropy values
print(entropy)


# In[9]:


# the entropy values (stored in the 'entropy' list)
# Create x-axis (position) values
positions = list(range(len(entropy)))
# Plot the entropy values
plt.figure(figsize=(14, 4))
plt.plot(positions, entropy, linestyle='-', color='b')
plt.xlabel("Position in Alignment")
plt.ylabel("Entropy")
plt.title("Shannon Entropy in Multiple Sequence Alignment")
plt.grid(True)
plt.show()


# In[10]:


#correl between my entropy and weblogo entropy
logofile = "./data/logo60_noadj.txt"  # or with our prevoius logo60seq file
dfweights0 = pd.read_csv(logofile, sep="\t", header=7) 
entropy_values_weblogo = dfweights0['Entropy']

from scipy.stats import pearsonr
import scipy.stats as ss
correlation_coefficient, p_value = ss.pearsonr(entropy_values_weblogo, entropy)
# Print the correlation coefficient
print(f"Pearson correlation coefficient: {correlation_coefficient:.7f}")
print(f"P-value: {p_value:.5f}")


# repository:

# In[11]:


# import math
# #Calculates the Shannon entropy for each position in a multiple sequence alignment.
# #alignment: A list of strings representing the sequences in the alignment.
# #valid_amino_acids: A string of valid amino acid characters (default: standard 20).
# #Returns: A list of entropy values, one for each position in the alignment.

# def calculate_entropy(alignment, valid_amino_acids="ACDEFGHIKLMNPQRSTVWY-"):
#     if not alignment:
#         raise ValueError("Empty alignment provided.")
#     num_sequences = len(alignment)
#     sequence_length = len(alignment[0])
#     entropy_values = [0] * sequence_length
#     amino_acid_indices = dict((aa, i) for i, aa in enumerate(valid_amino_acids))
#     for i in range(sequence_length):
#         amino_acid_counts = [0] * len(amino_acid_indices)
#         for sequence in alignment:
#             amino_acid = sequence[i]
#             if amino_acid in amino_acid_indices:
#                 amino_acid_counts[amino_acid_indices[amino_acid]] += 1
#             else:
#                 # Handle non-standard amino acids (e.g., log a warning, skip, etc.)
#                 pass
#         amino_acid_probabilities = [count / num_sequences for count in amino_acid_counts]
#         for probability in amino_acid_probabilities:
#             if probability > 0:
#                 entropy_values[i] += -probability * math.log(probability, math.e)  # OR log(,20)?
#     return entropy_values
# # Read the alignment from the file
# msafile = "./data/60seq.aln"
# alignment = AlignIO.read(msafile, "clustal")
# # Calculate the entropy values
# entropy = calculate_entropy(alignment)
# # Print the entropy values
# print(entropy)


# In[12]:


# # the entropy values (stored in the 'entropy' list)
# # Create x-axis (position) values
# positions = list(range(len(entropy)))
# # Plot the entropy values
# plt.figure(figsize=(14, 4))
# plt.plot(positions, entropy, linestyle='-', color='b')
# plt.xlabel("Position in Alignment")
# plt.ylabel("Entropy")
# plt.title("Shannon Entropy in Multiple Sequence Alignment")
# plt.grid(True)
# plt.show()


# In[13]:


# #correl between my entropy and weblogo entropy
# logofile = "./data/logo60_noadj.txt"  # or with our prevoius logo60seq file
# dfweights0 = pd.read_csv(logofile, sep="\t", header=7) 
# entropy_values_weblogo = dfweights0['Entropy']

# from scipy.stats import pearsonr
# import scipy.stats as ss
# correlation_coefficient, p_value = ss.pearsonr(entropy_values_weblogo, entropy)
# # Print the correlation coefficient
# print(f"Pearson correlation coefficient: {correlation_coefficient:.4f}")
# print(f"P-value: {p_value:.6f}")


# method2

# In[14]:


# # Calculates Shannon entropy for each position in a multiple sequence alignment.
# #Returns:A dictionary with keys as alignment column positions and values as entropy values.
# def calculate_entropy(alignment):
#   entropy_dict = {}
#   for col_index in range(alignment.get_alignment_length()):
#     col_data = alignment[:, col_index]
#     # Count amino acid frequencies
#     aa_counts = {}
#     for aa in col_data:
#       if aa not in aa_counts:
#         aa_counts[aa] = 0
#       aa_counts[aa] += 1
#     # Calculate total number of sequences
#     num_seqs = len(alignment)
#     # Calculate entropy
#     entropy = 0
#     for aa, count in aa_counts.items():
#       relative_freq = count / num_seqs
#       if relative_freq > 0:
#         entropy += relative_freq * math.log(relative_freq)
#     entropy *= -1  # Negate for convention as information content
#     entropy_dict[col_index] = entropy
#   return entropy_dict
# # Read alignment file
# alignment = AlignIO.read("data/60seq.aln", "clustal")
# # Calculate entropy
# entropy_dict = calculate_entropy(alignment)
# # Print entropy values
# for col_index, entropy in entropy_dict.items():
#   print(f"Position {col_index+1}: {entropy:.4f}")


# method3

# In[15]:


# def calculate_entropy(labels, base=None): # e - 2 - 10 - 20 - None
#     n_labels = len(labels)
#     if n_labels <= 1:
#         return 0
#     value, counts = np.unique(labels, return_counts=True)
#     probs = counts / n_labels
#     n_classes = np.count_nonzero(probs)
#     if n_classes <= 1:
#         return 0
#     ent = 0.
#     base = np.e if base is None else base
#     for i in probs:
#         ent -= i * np.log(i) / np.log(base)
#     return ent
# # Read MSA from file (replace with your MSA file path)
# msafile = "./data/60seq.aln"
# alignment = AlignIO.read(msafile, "clustal")
# # Calculate entropy for each position in the MSA
# msa_entropy = []
# for i in range(alignment.get_alignment_length()):
#     column = alignment[:, i]
#     labels = [seq.upper() for seq in column]
#     msa_entropy.append(calculate_entropy(labels))
# # Print entropy values for each position
# for i, entropy_value in enumerate(msa_entropy):
#     print(f"Position {i}: Entropy = {entropy_value:.4f}")


# In[16]:


# # entropy values(stored in the 'msa_entropy' list)
# # Create x-axis (position) values
# positions = list(range(len(msa_entropy)))
# # Plot the entropy values
# plt.figure(figsize=(14, 4))
# plt.plot(positions, msa_entropy, linestyle='-', color='b')
# plt.xlabel("Position in Alignment")
# plt.ylabel("Entropy")
# plt.title("Shannon Entropy in Multiple Sequence Alignment")
# plt.grid(True)
# plt.show()


# In[17]:


# # msa_entropy
# correlation_coefficient, _ = pearsonr(entropy_values_weblogo, msa_entropy)
# # Print the correlation coefficient
# print(f"Pearson correlation coefficient: {correlation_coefficient:.4f}")
# print(f"P-value: {p_value:.6f}")


# method4

# In[18]:


# # Replace with your actual file path ...another way for entropy
# msa_file = "./data/60seq.aln"
# # Read the MSA
# alignment = AlignIO.read(msa_file, "clustal")
# # Get the number of sequences and alignment length
# num_sequences = len(alignment)
# alignment_length = len(alignment[0])
# # Calculate Shannon entropy at each position
# entropy_scores = []
# for i in range(alignment_length):
#     # Count nucleotide frequencies at position i
#     nucleotide_counts = {nt: 0 for nt in "ACDEFGHIKLMNPQRSTVWY-"}
#     for seq in alignment:
#         nucleotide_counts[seq[i]] += 1
#     # Calculate probabilities and Shannon entropy
#     probabilities = [count / num_sequences for count in nucleotide_counts.values()]
#     entropy_scores.append(sum(-p * math.log2(p) for p in probabilities if p > 0))
# # Print or analyze the entropy scores for each position
# print(entropy_scores)


# In[19]:


# correlation_coefficient, _ = pearsonr(entropy_values_weblogo, entropy_scores)
# # Print the correlation coefficient
# print(f"Pearson correlation coefficient: {correlation_coefficient:.4f}")
# print(f"P-value: {p_value:.6f}")


# ## Weight

# In[20]:


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
msafile = "./data/60seq.aln"
column_weights = calculate_weights(msafile)
# Print the weights for each column
# for i, weight in enumerate(column_weights, start=1):
#     print(f"Column {i}: Weight = {weight:.4f}")


# In[21]:


plt.figure(figsize=(14, 4))
plt.bar(range(1, len(column_weights) + 1), column_weights)
plt.xlabel("Column Number")
plt.ylabel("Weight")
plt.title("Weights for Each Column in MSA")
plt.grid(axis="y", linestyle="--", alpha=0.7)
# Show the plot
plt.tight_layout()
plt.show()


# In[22]:


# # with padding "-" 
# msafile = "./data/60seq.aln"
# alignment = AlignIO.read(msafile, "clustal")
# sequences = [record.seq for record in alignment] # Extract the sequences from the alignment
# sequence_list = [str(seq) for seq in sequences] # Create a list of sequence letters
# max_length = max(len(seq) for seq in sequence_list) # Determine the maximum sequence length
# # Create a matrix (list of lists) with padding for shorter sequences
# sequence_matrix = []
# for seq in sequence_list:
#     padded_seq = seq.ljust(max_length, "-")  # Pad with hyphens if needed
#     sequence_matrix.append([char for char in padded_seq])  # Each sequence as a separate row
# # Print the sequence matrix
# # for row in sequence_matrix:
# #     print(" ".join(row))
# counts_df = lm.alignment_to_matrix(sequences=sequence_list, to_type='counts') #, characters_to_ignore='.-X'
# counts_df


# In[23]:


# lm.Logo(counts_df,figsize=(14,4))


# In[24]:


# # filter based on counts example
# num_seqs = counts_df.sum(axis=1)
# pos_to_keep = num_seqs > len(sequence_list)/2
# ww_counts_df = counts_df[pos_to_keep]
# ww_counts_df.reset_index(drop=True, inplace=True)
# lm.Logo(ww_counts_df)


# In[25]:


# # filtered example
# ww_counts_df = counts_df.iloc[500:551]
# # Reset the index to start from 0
# ww_counts_df.reset_index(drop=True, inplace=True)
# lm.Logo(ww_counts_df)


# In[27]:


# padding idea removed
msafile = "./data/60seq.aln"
alignment = AlignIO.read(msafile, "clustal")
sequences = [str(record.seq) for record in alignment] # Extract the sequences from the alignment
sequence_list = [str(seq) for seq in sequences] # Create a list of sequence letters
# Determine the length of the longest sequence
max_length = max(len(seq) for seq in sequences)
# Create a matrix (list of lists) without padding
sequence_matrix = [list(seq) for seq in sequences]
# If you still need counts_df, you can proceed with it
counts_df = lm.alignment_to_matrix(sequences=sequence_list, to_type='counts') #, characters_to_ignore='.-X'
# Continue with the rest of your code
counts_df


# In[28]:


lm.Logo(counts_df,figsize=(14,4))


# In[ ]:




