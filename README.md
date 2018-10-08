# An Improved Deep Embedding Learning Method for Short Duration Speaker Verification -  Pytorch Implementation

This is a pytorch implementation of the model(modified cross-conv. pooling) presented by Zhifu Gao in [An Improved Deep Embedding Learning Method for Short Duration Speaker Verification](https://kar.kent.ac.uk/67451/).

I am sorry that most of the code except the model is old and dirty. Because I try to it only private database. but there is no problem with performance or operation. If you only fit the input size - batch X 1 X feature dim. X frame.

Original paper's parameter is very big model. Cross-conv. pooling layer output is 512 x 512 = 262144, it makes small batch size and a lot of training time and so on. I recommend you use small size parameter about 128 x 128.

I hope this code helps researcher reach higher score.

## Data input
 - batch X 1 X feature dim. X frame.

## Credits
Original paper:
- Gao's paper:
```
@article{,
  author    = {Zhifu Gao, Yan Song, Ian McLoughlin, Wu Guo and Lirong Dai},
  title     = {An Improved Deep Embedding Learning Method for Short Duration Speaker Verification},
  conference   = {Interspeech 2018},
  year      = {2018},
}
```

Also, use the part of code:
- [my git repository](https://github.com/qqueing/DeepSpeaker-pytorch)
   - Baseline code - data loader and so on.
- [liorshk's git repository](https://github.com/liorshk/facenet_pytorch)
   - Facenet pytorch implimetation
- [hbredin's git repository](https://github.com/hbredin/pyannote-db-voxceleb)
   - Voxceleb Database reader


## Features
 - This code has only model implementation. Data loader and the other code was recycled from [this code](https://github.com/qqueing/DeepSpeaker-pytorch)


## Authors
qqueing@gmail.com( or kindsinu@naver.com)

