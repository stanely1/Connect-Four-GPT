"""
Prepare the Connect-Four dataset for character-level language modeling.
Each token represents a single move in the game or is a special token
representing start of the game (S) or the game result (A/B/D).
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np

# read data
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r') as f:
    # data = f.read().strip().split('\n')
    data = f.read().replace('\n', '')
# print(f"length of dataset in games: {len(data):,}")
print(f"length of dataset in characters: {len(''.join(data)):,}")
# print(f"max game length: {max(len(d) for d in data)}")

# vocab
chars = '0123456ABDS'
vocab_size = len(chars)
print("all the unique characters:", chars)
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
split_n = int(n*0.9)
while data[split_n] != 'S':
    split_n -= 1
train_data = data[:split_n]
val_data = data[split_n:]

# encode both to integers
# train_ids = [encode(g) for g in train_data]
# val_ids = [encode(g) for g in val_data]
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 738,699
# all the unique characters: 0123456ABDS
# vocab size: 11
# train has 664,824 tokens
# val has 73,875 tokens
