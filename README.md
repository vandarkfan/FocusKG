﻿<h1 align="center">
  DTME
</h1>

<h4 align="center">FocusKG: A Novel Multimodal Knowledge Graph Dataset with Temporal Information for Link Prediction</h4>

<h2 align="center">
  Overview of DTME
  <img align="center"  src="overview.png" alt="...">
</h2>


This paper has been submitted to the TKDE.

### Dependencies

- python            3.10.13
- torch             2.1.1+cu118
- numpy             1.26.3
- transformers      4.44.1

### Dataset:

- We constructe a new discrete temporal multimodal knowledge graph dataset: FocusKG. 
- We use WN18RR++, FB15K237, and VTKG-I dataset for knowledge graph link prediction. 

### Results:
The results are:

| Dataset  | MRR  | H@1  | H@3  | H@10 |
| :------: | :--: | :--: | :--: | :--: |
| FocusKG  | 41.5 | 31.8 | 49.0 | 63.0 |
| WN18RR++ | 55.6 | 48.9 | 57.8 | 67.3 |
| FB15K237 | 37.1 | 27.6 | 40.1 | 55.7 |
|  VTKG-I  | 48.7 | 40.8 | 50.4 | 63.7 |

## How to Run
```
python train --data FocusKG --lr 0.0008 --dim 512 --num_epoch 750 --valid_epoch 50 --exp best --hidden_dim 2048 --dropout 0.1 --emb_dropout 0.5 --img_dropout 0.1 --txt_dropout 0.5 --vid_dropout 0.5 --aud_dropout 0.2 --time_dropout 0.7 --smoothing 0.0 --batch_size 1024 --decay 0.0 --step_size 50   ## FocusKG



python train --data WN18RR++ --lr 0.001 --dim 256 --num_epoch 750 --valid_epoch 50 --exp best --hidden_dim 2048 --dropout 0.1 --emb_dropout 0.6 --img_dropout 0.3 --txt_dropout 0.4 --vid_dropout 0 --aud_dropout 0 --time_dropout 0 --smoothing 0.0 --batch_size 1024 --decay 0.0 --step_size 50   ## WN18RR++



python train --data FB15K237 --lr 0.0001 --dim 256 --num_epoch 750 --valid_epoch 50 --exp best --hidden_dim 2048 --dropout 0.1 --emb_dropout 0.8 --img_dropout 0.4 --txt_dropout 0.1 --vid_dropout 0 --aud_dropout 0 --time_dropout 0 --smoothing 0.0 --batch_size 512 --decay 0.0 --step_size 50   ## FB15K237



python train --data VTKG-I --lr 0.001 --dim 256 --num_epoch 150 --valid_epoch 50 --exp best --hidden_dim 2048 --dropout 0.1 --emb_dropout 0.7 --img_dropout 0.5 --txt_dropout 0.6 --vid_dropout 0 --aud_dropout 0 --time_dropout 0 --smoothing 0.0 --batch_size 128 --decay 0.0 --step_size 50   ## VTKG-I
```

