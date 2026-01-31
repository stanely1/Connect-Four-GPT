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
    data = f.read().strip().split('\n')
print(f"length of dataset in games: {len(data):,}")
print(f"length of dataset in characters: {len(''.join(data)):,}")

max_game_len = max(len(d) for d in data)
print(f"max game length: {max_game_len}")

# vocab
chars = '0123456ABDS'
vocab_size = len(chars)
print("all the unique characters:", chars)
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers and characters to moves
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
stom = { ch:ch for ch in chars}
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
split_n = int(n*0.9)
train_data = data[:split_n]
val_data = data[split_n:]

# encode both to integers
train_ids = [encode(g) for g in train_data]
val_ids = [encode(g) for g in val_data]
print(f"train has {len(train_ids):,} games ({sum(len(g) for g in train_ids)} tokens)")
print(f"val has {len(val_ids):,} games ({sum(len(g) for g in val_ids)} tokens)")

# align length of each game to max - duplicate last token
# using max_game_len + 1 to simplify training:
# e.g:
# original: S1123D
# extended: S1123DDD
# x: S1123DD (game[:-1])
# y: 1123DDD (game[1:])
for g in train_ids: g += [g[-1]] * (max_game_len + 1 - len(g))
for g in val_ids: g += [g[-1]] * (max_game_len + 1 - len(g))

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
eos_chars = 'ABD'
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'stom': stom,
    'block_size': max_game_len,
    'eos_token_ids': encode(eos_chars),
    'start_token': 'S',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in games: 20,000
# length of dataset in characters: 738,699
# max game length: 44
# all the unique characters: 0123456ABDS
# vocab size: 11
# train has 18,000 games (664297 tokens)
# val has 2,000 games (74402 tokens)
