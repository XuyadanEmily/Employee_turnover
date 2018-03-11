# Employee_turnover
1-->run Employee_turnover.py第一版，发现两个问题：一是在数据的选取上，因为正负样本的数量比例不等，所以在Train-test测试过程中，直接进行28比例分割，可能导致在训练数据中正样本数量太少达不到训练的效果；二是binary数据不能直接用于LR的问题，后续需要解决这个问题；
