### HW5: Matrix factorization and DNN method
* Library: argparse,pandas,numpy,keras,pickle,sklearn,matplotlib 
* Training using MF method: ml2017fallhw5_mf-keras.py\
  usage: ml2017fallhw5_mf-keras.py [-h] [--user_file USERF]
                                 [--movie_file MOVIEF] [--train_file TRAINF]
                                 [--isnorm] [--latent_dim LATENT_DIM]
                                 [--savetrain]

* Training using DNN method: ml2017fallhw5_dnn-keras.py\
  usage: ml2017fallhw5_dnn-keras.py [-h] [--user_file USERF]
                                  [--movie_file MOVIEF] [--train_file TRAINF]
                                  [--isnorm] [--latent_dim LATENT_DIM]
                                  [--savetrain]
                                  
* Predict Rates for test data: ml2017fallhw5_mftest-keras.py\
  usage: ml2017fallhw5_mftest-keras.py [-h] [--test_file TESTF] [--isnorm]
                                     [--modelf MODELF] [--test_submit TESTSUB]
                                     
* Drawing 2-dim projection by t-SNE: ml2017fallhw5_mf_tsne.py\
  usage: ml2017fallhw5_mftest-keras.py [-h] [--movie_file MOVIEF]
                                     [--movie_emb M_EMB]
                                     
## Loss (RMSE) verse epochs
* MF: The loss became stable from ~100 epoches. The best loss (RMSE) of MF method with normalized rates is about 0.964.

* DNN: The loss became stable from ~15 epoches. The best loss (RMSE) of DNN method is about 0.89.

* This plot shows the comparison between DNN and MF methods

#![Loss comparison](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw5/result/loss_compare3.png "MF & DNN RMSE")

## T-SNE for movie label
#![movie Geren labels](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw5/tsne_dnn_50d.png "TSNE plot for movie Genres labels")
