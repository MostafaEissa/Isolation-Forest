## Isolation Forest

This is a demo to show my implementation of isolation forest based on [this paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

> F. T. Liu, K. M. Ting and Z. Zhou, "Isolation Forest," 2008 Eighth IEEE International Conference on Data Mining, Pisa, 2008, pp. 413-422.

Isolation forests are used for anamoly detection and the idea is based on the observation that anomalies have distinctive quantitative properties:

1. they are the minority consisting of fewer instances
2. they have attribute-values that are very different from those of normal instances