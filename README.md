## Cost-Sensitive Multi-Label Classification 

Python implementation of three cost-sensitive multi-label classification algorithms and two general multi-label algorithms.

- Cost-Sensitive Label Embedding with Multidimensional Scaling (CLEMS)

  Kuan-Hao Huang and Hsuan-Tien Lin.
  Cost-Sensitive Label Embedding for Multi-Label Classification.
  Machine Learning, 2017

- Condensed Filter Tree (CFT)

  Chun-Liang Li and Hsuan-Tien Lin.
  Condensed Filter Tree for Cost-Sensitive Multi-Label Classification.
  ICML, 2014.

- Probabilistic Classifier Chains (PCC)

  Krzysztof Dembczynski, Weiwei Cheng, and Eyke Hullermeier.
  Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains.
  ICML, 2012.
  
- Classifier Chains (CC)

  Jesse Read, Bernhard, Pfahringer, Geoff Holmes, and Eibe Frank.
  Classifier chains for multi-label classification.
  Machine Learning, 2011

- Binary Relevance (BR)

  Grigorios Tsoumakas and Ioannis Katakis.
  Multi-Label Classification: An Overview.
  International Journal of Data Warehousing and Mining, 2007.


### Prerequisites 
- Python 2.7.12
- NumPy 1.13.3
- scikit-learn 0.17

### Usage 

    $ python demo.py
    
### Dataset

- scene (downloaded from [Mulan](http://mulan.sourceforge.net/datasets-mlc.html))

### Evaluation Criteria

- Hamming loss
- Rank loss
- F1 score
- Accuracy score

### Result

    ============================================================
    algorithm  hamming_loss  rank_loss  f1_score  accuracy_score
    ============================================================
           BR        0.0907     1.1844    0.5742          0.5627
           CC        0.0880     1.1424    0.5947          0.5851
          PCC        0.0900     0.6898    0.7460          0.6909
          CFT        0.0867     0.9460    0.6478          0.6267
        CLEMS        0.0825     0.6553    0.7690          0.7600
    ============================================================

    

### Author

Kuan-Hao Huang / [@ej0cl6](http://ej0cl6.github.io/)
