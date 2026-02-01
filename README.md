
# Connect-Four-GPT

A simple GPT model trained to predict moves in the game [Connect-Four](https://en.wikipedia.org/wiki/Connect_Four). This project is based on [nanoGPT](https://github.com/karpathy/nanoGPT).


## Install dependencies

In main project directory run

```
pip install -r requirements.txt
```

## Prepare dataset

Datasets are contained in `data` directory. Available variants:

- `connect_four_simple`: token represents only a column where a piece is inserted.
- `connect_four_player`: token represents a column and the player who performs a move.
- `connect_four_full`: token represents column, row and player.

For selected dataset run: (example for `connect_four_simple`, you can replace it with any of the above)

```sh
python data/connect_four_simple/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. This data will be used to train the model.

To train a model run:

```sh
python train.py config/train_connect_four_simple.py
```

When the model is ready, you can use `sample.py` script to draw samples from it:

```sh
python sample.py --out_dir=out-connect_four_simple
```

## Experiments

### What was tested

Model output:

- How often the model predicts legal moves?
- How often it predicts the correct game result?

Probing from internal layers:

- Does the model learn board state representation?
- How it represents the value of each cell?
- How it represents the number of pieces in each row/column?

### Results

#### Legal move and game result prediction

##### `connect_four_simple`:
- Legal moves predicted (training data): 495689/498639 (99.41%)
- Game result predicted (training data): 16097/18000 (89.43%)
- Legal moves predicted (validation data): 59234/59665 (99.28%)
- Game result predicted (validation data): 1691/2000 (84.55%)

##### `connect_four_player`
- Legal moves predicted (training data): 495395/498639 (99.35%)
- Game result predicted (training data): 16244/18000 (90.24%)
- Legal moves predicted (validation data): 59198/59665 (99.22%)
- Game result predicted (validation data): 1676/2000 (83.80%)

##### `connect_four_full`
- Legal moves predicted (training data): 497649/498639 (99.80%)
- Game result predicted (training data): 17053/18000 (94.74%)
- Legal moves predicted (validation data): 59454/59665 (99.65%)
- Game result predicted (validation data): 1800/2000 (90.00%)

#### Probing

For verification of probing results the experiments were also performed for untrained model.
This model was created with
```
python train.py config/train_connect_four_simple.py --out_dir=out-connect-four-untrained --max_iters=0 --eval_only=True --save_initial_checkpoint=True
```

##### Control test (untrained model):

![](assets/control_cell_train.png)

![](assets/control_cell_val.png)

![](assets/control_col_train.png)

![](assets/control_col_val.png)

![](assets/control_row_train.png)

![](assets/control_row_val.png)

##### `connect_four_simple`:

![](assets/simple_cell_train.png)

![](assets/simple_cell_val.png)

![](assets/simple_col_train.png)

![](assets/simple_col_val.png)

![](assets/simple_row_train.png)

![](assets/simple_row_val.png)

##### `connect_four_player`

![](assets/player_cell_train.png)

![](assets/player_cell_val.png)

![](assets/player_col_train.png)

![](assets/player_col_val.png)

![](assets/player_row_train.png)

![](assets/player_row_val.png)

##### `connect_four_full`

![](assets/full_cell_train.png)

![](assets/full_cell_val.png)

![](assets/full_col_train.png)

![](assets/full_col_val.png)

![](assets/full_row_train.png)

![](assets/full_row_val.png)
