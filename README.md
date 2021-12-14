# Target-Based-Sentiment-Analysis

LSTM's based Neural Network Classification Models

### Task 

Given a text and a *"*phrase*"* from it, detect the sentiment expressed towards the *"phrase"* in the text instance.

### Approach 
The approach and design of the architecture is inspired from the research paper - **[Effective LSTMs for Target-Dependent Sentiment Classification](https://arxiv.org/abs/1512.01100)**

### Accuracy 

The designed model achieved 86% accuracy, which ran for 100 epochs on 4500 training examples.

### Remark

The architecutre is efficient because it reached 86 % accuracy on train dataset which consists of only 4500 examples and the evaluation metrics suggests that there is no overfit of the model.  

### Requirements 

Python 3 (Tested on python 3.7.12) + Tensorflow (Tested using version 2.7)

### Source code tree

```
.
├── data
│   ├── embeddings.txt
│   ├── preprocessed_train.txt
│   ├── results
│   │   └── results.csv
│   ├── test.csv
│   ├── train.csv
│   └── word2Idx.txt
├── LICENSE
├── models
│   ├── saved_models
│   │   └── trained_model.h5
│   └── train_LSTM.py
├── notebooks
├── README.md
└── src
    ├── evaluation.py
    ├── preprocessing.py
    └── utils.py
```
