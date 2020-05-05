# ToyML
Learning Machine Learning with toy code.

There are some machine learning algorithms implemented from scratch!

## Overview

这是一个已经进行了很久，并将长期进行下去的项目。在之前的项目[ToyData](https://github.com/shenxiangzhuang/ToyData)中，我手动实现了一些常见的数据结构并将其封装成库，写了单元测试，简单的文档，在这整个过程中受益良多。

这个项目旨在加深对于机器学习的理解，因为很久之前就意识到写在书上的公式和代码之前存在着一条“鸿沟”——至少对于我这种代码水平比较差的人来说是这样。这条“鸿沟”也是“看懂了”与“学会了”之间的鸿沟，我想做的就是尽可能填补上这一空白。

## Intro

项目距离封装成库还有很远的路要走，目前的完成度远远不够...核心的代码全部放在`myml`文件夹下，源码总体分为两个部分：算法应用Demo和手写算法。

算法应用Demo是以`ipynb`的形式存在，是调用算法的一些示例程序，编程语言涵盖`Python`和`R`, 库尽量使用主流的机器学习库。这些调用的算法程序完成比较高，涵盖大部分经典的算法,如下:

- [x] Association Analysis: Apriori, Relim
- [x] Clutering: DBSCAN, Hierarchical(Agnes&Diana), Kmeans
- [x] Classification: SVM, NaiveBayes, KNN, DecisionTree(ID3, C4.5, CART, C50)
- [x] Ensemble: Boosting, Bagging, Stacking

手写算法就是用纯`Python`来写算法的实现，为了明晰每一步的运算，我在写的过程中会尽量减少`Numpy`的使用。

- [x] Clutering: DBSCAN, Hierarchical(Agnes&Diana), Kmeans
- [x] Classification: KNN
- [x] Ensemble: Boosting(AdaBoost)

下一步的计划
- [ ] Classification: NaiveBayes, DecisionTree, SVM
- [ ] Association Analysis: Apriori
- [ ] Ensemble: GBDT

## And
从理论到实现，从实现到更高效的实现，从高效的实现到高效易读的实现...这将是一个漫长的`commit`过程，这个过程必然没有终点，但有一点毫无疑问，那就是我们将越走越远.
