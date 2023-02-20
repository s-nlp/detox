# Detoxification
Detoxification is an automatic transformation of a text such that:
- text becomes non-toxic
- the content of the text stays the same.

This repository contains the code and data for the paper "[Text Detoxification using Large Pre-trained Neural Models](https://aclanthology.org/2021.emnlp-main.629/)" ([video](https://youtu.be/knhIJ_e9Uyc)).

We suggest two models:
- **CondBERT** — a BERT-based model which identifies toxic words in a text and replaces them with neutral synonyms
- **ParaGeDi** — a paraphraser-based model which re-generates a text using additional style-informed LMs

**If you have any questions about the models, the code, or the data, please do not hesitate to communicate via [GitHub issues](https://github.com/skoltech-nlp/detox/issues)!**.

If you want the fastest way to run the inference of these models, you can run 
[this Colab notebook](https://colab.research.google.com/drive/1HEUgKO100rSuDcv80prD3dRZSjKztTqk?usp=sharing) 
that puts together some of the code from this repository. 

## CondBERT

The notebooks for reproducing the training and inference of this model in the folder [condBERT](/emnlp2021/style_transfer/condBERT).

## ParaGeDi

The notebooks and scripts for reproducing the training and inference of this model in the folder [paraGeDi](/emnlp2021/style_transfer/paraGeDi).

## Parallel detoxification corpus

The notebooks for reproducing the data collection and training the model on it are in the folder [mining_parallel_corpus](/emnlp2021/style_transfer/mining_parallel_corpus).

The original ParaNMT corpus (50M sentence pairs) can be downloaded from the authors page: https://www.cs.cmu.edu/~jwieting/. 
The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip).

The paraphraser trained on this filtered corpus is available at https://huggingface.co/s-nlp/t5-paranmt-detox. 

## Evaluation

To evaluate your model, use the folder [metric](/emnlp2021/metric). 

First, download the models for content preservation and fluency with the script `prepare.sh`. 

Then run the script `metric.py`, as in the example below:

```
python metric/metric.py --inputs data/test/test_1ok_toxic --preds data/test/model_outputs/condbert.txt
```

## Acknowledgements
This research was conducted under the framework of the [Joint MTS-Skoltech laboratory](https://sites.skoltech.ru/nlp/joint-mts-skoltech-laboratory/). 
We are grateful to the reviewers for their helpful suggestions which substantially improved this work.


## Citation

If you use our models or data, please cite the paper:

```
@inproceedings{dale-etal-2021-text,
    title = "Text Detoxification using Large Pre-trained Neural Models",
    author = "Dale, David  and
      Voronov, Anton  and
      Dementieva, Daryna  and
      Logacheva, Varvara  and
      Kozlova, Olga  and
      Semenov, Nikita  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.629",
    pages = "7979--7996",
}
```
