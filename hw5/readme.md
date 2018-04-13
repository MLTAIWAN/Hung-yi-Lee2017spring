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
                                     
## Loss of Matrix Factorization 
* The loss became stable from ~100 epoches. The best loss (RMSE) of MF method with normalized rates is about 0.964.

#![Loss MF Latent dimension = 10](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw5/result/mfloss_rmse_d10.png "MF RMSE")

## Loss of DNN
* The loss became stable from ~15 epoches. The best loss (RMSE) of DNN method is about 0.89.

#![Loss DNN Latent dimension = 15](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw5/result/dnnloss_rmse_norm_d15.png "DNN RMSE")
* This plot shows the comparison between DNN and MF methods

#![Loss comparison](https://github.com/MLTAIWAN/Hung-yi-Lee2017spring/blob/kunxian/hw5/result/loss_compare3.png "MF & DNN RMSE")
