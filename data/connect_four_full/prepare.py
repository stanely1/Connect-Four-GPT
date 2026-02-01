'''
Prepare the Connect-Four dataset for character-level language modeling.
Each token represents a single move in the game or is a special token
representing start of the game (S) or the game result (A/B/D).
Moves encode column, row and player (e.g. 15a 33b ...).
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
'''
import os
import pickle
import numpy as np

# read data
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r') as f:
    data = f.read().strip().split('\n')

separator = ' '
data = [d.split(separator) for d in data]

print(f'length of dataset in games: {len(data):,}')
print(f'length of dataset in moves: {sum(len(d) for d in data):,}')

max_game_len = max(len(d) for d in data)
print(f'max game length: {max_game_len}')

# vocab
tokens = 'S A B D'.split() + [x + y + p for x in '0123456' for y in '012345' for p in 'ab']
vocab_size = len(tokens)
print('all the unique tokens:', ' '.join(tokens))
print(f'vocab size: {vocab_size:,}')

# create a mapping from tokens to integers and tokens to moves
stoi = { t:i for i,t in enumerate(tokens) }
itos = { i:t for i,t in enumerate(tokens) }
stom = { t:t[0] for t in tokens}

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return separator.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
split_n = int(n*0.9)
train_data = data[:split_n]
val_data = data[split_n:]

# encode both to integers
train_ids = [encode(g) for g in train_data]
val_ids = [encode(g) for g in val_data]
print(f'train has {len(train_ids):,} games ({sum(len(g) for g in train_ids):,} tokens)')
print(f'val has {len(val_ids):,} games ({sum(len(g) for g in val_ids):,} tokens)')

# align length of each game to max - duplicate last token
# using max_game_len + 1 to simplify training:
# e.g:
# original: S 15a 14b 25a 35b D
# extended: S 15a 14b 25a 35b D D D
# x:   S 15a 14b 25a 35b D D (game[:-1])
# y: 15a 14b 25a 35b D   D D (game[1:])
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
    'separator': separator,
    'block_size': max_game_len,
    'eos_token_ids': encode(eos_chars),
    'start_token': 'S',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in games: 20,000
# length of dataset in moves: 738,699
# max game length: 44
# all the unique tokens: S A B D 00a 00b 01a 01b 02a 02b 03a 03b 04a 04b 05a 05b 10a 10b 11a 11b 12a 12b 13a 13b 14a 14b 15a 15b 20a 20b 21a 21b 22a 22b 23a 23b 24a 24b 25a 25b 30a 30b 31a 31b 32a 32b 33a 33b 34a 34b 35a 35b 40a 40b 41a 41b 42a 42b 43a 43b 44a 44b 45a 45b 50a 50b 51a 51b 52a 52b 53a 53b 54a 54b 55a 55b 60a 60b 61a 61b 62a 62b 63a 63b 64a 64b 65a 65b
# vocab size: 88
# train has 18,000 games (664,297 tokens)
# val has 2,000 games (74,402 tokens)
