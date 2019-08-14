# Task-oriented Dialogue with GECOR

Source code for the EMNLP 2019 paper entitled "GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue" by Jun Quan et al.


```
"GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue" 
(Jun Quan, Deyi Xiong, Bonnie Webber and Changjian Hu. EMNLP, 2019)

We will soon upload the bibtex.

```


## Training with default parameters

```
python model.py -mode train -model gecor-camrest
```

(optional: configuring hyperparameters with cmdline)

```
python model.py -mode train -model gecor-camrest -cfg lr=0.003 batch_size=32
```

## Testing

```
python model.py -mode test -model gecor-camrest
```

