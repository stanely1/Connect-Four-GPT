'''
Play a game with GPT model
'''
import os
import pickle
import sys
import torch
import numpy as np
from c4engine import C4Engine
from model import GPTConfig, GPT


if __name__ == '__main__':
    rng = np.random.default_rng()

    DEVICE = 'cuda'

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)

    out_dir = sys.argv[1]

    # load model
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    # read metadata
    data_dir = os.path.join('data', checkpoint['config']['dataset'])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # get encode/decode - tokenizer
    vocab_size = meta['vocab_size']
    stoi, itos, stom, separator = meta['stoi'], meta['itos'], meta['stom'], meta['separator']
    split = lambda s: list(s) if separator == '' else s.split(separator)
    encode = lambda s: [stoi[c] for c in split(s)]
    decode = lambda l: separator.join(itos[i] for i in l)
    ids_to_moves = lambda l: ''.join(stom[itos[i]] for i in l)

    # TODO: handle it cleaner, maybe add this mapping to meta
    def simple_moves_to_model_encoding(moves: str):
        if vocab_size == 11: # simple model
            return encode(moves)
        if vocab_size == 18: # player model
            result_moves = []
            for i,m in enumerate(moves):
                if m in C4Engine.START + C4Engine.RESULT_TYPES:
                    result_moves.append(m)
                else:
                    result_moves.append(f'{m}{'a' if i % 2 == 1 else 'b'}')
            return encode(separator.join(result_moves))
        if vocab_size == 88: # full model
            result_moves = []
            eng = C4Engine()
            for i,m in enumerate(moves):
                eng.make_move(m)
                if m in C4Engine.START + C4Engine.RESULT_TYPES:
                    result_moves.append(m)
                else:
                    result_moves.append(f'{m}{eng._last_y}{'a' if i % 2 == 1 else 'b'}')
            return encode(separator.join(result_moves))

    # game loop
    while True:
        prefix = 'S'
        engine = C4Engine(prefix)
        player = 'A' if rng.random() < 0.5 else 'B'

        print('You play as', player)

        while engine.result() is None:
            curr_player = engine.player_to_move()
            if curr_player == player:
                while True:
                    move = input(f'Your move ({player}): ')
                    if engine.make_move(move):
                        prefix += move
                        print('\n'.join(''.join(row) for row in engine.board()))
                        break
                    else:
                        print('This move is illegal!')
                        continue
            else:
                x = (torch.tensor(simple_moves_to_model_encoding(prefix), dtype=torch.long, device=DEVICE)[None, ...])
                y = model.generate(idx=x, max_new_tokens=1, top_k=1)
                move = ids_to_moves(y[0].tolist())[-1]
                if engine.make_move(move):
                    prefix += move
                    print(f'Player {curr_player} (GPT): {move}')
                    print('\n'.join(''.join(row) for row in engine.board()))
                else:
                    print(f'Player {curr_player} (GPT) attempted to play an illegal move: {move}')
                    engine._turn = 1 - engine._turn

        restart = input(f'Game finished with result: {engine.result()}. Do you want to play again? (y/n) ')
        if restart.lower() != 'y':
            break
