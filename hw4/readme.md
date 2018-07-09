
### Text Sentiment Classification
* Hung-yi Lee ML class HW4
* Library: argparse,collection,itertools,pandas,numpy,tensorflow 

## usage
* Wording Embedding: wordembed_cbow.py (CBOW method, will add skim-gram method later)
usage: wordembed_cbow.py [-h] [--vocab_size VOCAB_SIZE]
                         [-emb_dim EMBEDDING_DIM]
                         [--gpu_fraction GPU_FRACTION]
                         [--skip_window SKIP_WINDOW] [--num_skips NUM_SKIPS]
                         [--batch_size BATCH_SIZE]
                         [--learning_rate LEARNING_RATE] [--log_dir LOG_DIR]
                         [--nsteps NSTEPS] [--load_model LOAD_MODEL]
                         [--load_token LOAD_TOKEN] [--save_embed SAVE_EMBED]

* Sentiment Classification: ml2017fallhw4_wordem.py (the embedding matrix can be import and assign to model)
usage: ml2017fallhw4_wordem.py [-h] [--batch_size BATCH_SIZE]
                               [--nb_epoch NB_EPOCH] [--val_ratio VAL_RATIO]
                               [--gpu_fraction GPU_FRACTION]
                               [--vocab_size VOCAB_SIZE]
                               [--max_length MAX_LENGTH] [--patience PATIENCE]
                               [--loss_function LOSS_FUNCTION]
                               [--cell {LSTM,GRU}] [-num_lay NUM_LAYERS]
                               [-emb_dim EMBEDDING_DIM] [-hid_siz HIDDEN_SIZE]
                               [--pretrain_emb PRETRAIN_EMB]
                               [--emb_matrix EMB_MATRIX]
                               [--keep_prob KEEP_PROB] [-lr LEARNING_RATE]
                               [--threshold THRESHOLD]
                               [--result_path RESULT_PATH]
                               [--load_model LOAD_MODEL]
                               [--load_token LOAD_TOKEN] [--save_dir SAVE_DIR]
                               [--log_dir LOG_DIR]
                               model {train,test,semi}

Sentiment classification

positional arguments:
  model
  {train,test,semi}

## NN architecture
* picture of 3 layers of LSTM RNN
#![RNN architeture](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw4/plots/architeture.jpg "3 layers LSTM")

## Cost of training and validation at training
* picture of cost by tensorboard
#![cost](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw4/plots/cost_trainvalid.png "cost function during training")

## Accuraycy of validation at training
* picture of accuracy for validation data
#![accuracy](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw4/plots/accuracy_valid.png "accuracy function during training")
