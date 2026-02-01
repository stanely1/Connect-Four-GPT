# train a miniature character-level connect-four model

out_dir = 'out-connect-four-player'
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'connect-four-player'
wandb_run_name = 'mini-gpt'

dataset = 'connect_four_player'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 44 # context size - max game length (42 moves + start + result tokens)

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 2500
lr_decay_iters = 2500
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
