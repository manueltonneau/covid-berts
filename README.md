# Covid-BERTs

This repository contains information on two BERT versions pretrained on a preprocessed version of the [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset, namely **ClinicalCovidBERT** and **BioCovidBERT**. 

![Illustration](clash_covid.png)


## Contribution

 This project was inspired by the `covid_bert_base` [model](https://huggingface.co/deepset/covid_bert_base) from Deepset and [discussions](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/138250) on potential improvements of this model on Kaggle. My contribution is based on two pillars:
- better initialization: we initialize the training with existing BERT versions trained on scientific corpuses, namely [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) and [BioBERT](https://github.com/dmis-lab/biobert).
- specialized vocabulary: for one of the two models, we also use a customized vocabulary provided on the [BioBERT repository](https://github.com/dmis-lab/biobert) and use for training [BioBERT-Large v1.1 (+ PubMed 1M)](https://github.com/dmis-lab/biobert)

## Download

| Model                            | Downloads
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `clinicalcovid_bert_base_cased`   | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/clinicalcovid_bert/bert_config.json) • [`tensorflow weights`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/clinicalcovid_bert/clinicalcovid_bert_base_cased.ckpt.data-00000-of-00001) • [`tensorflow.meta`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/clinicalcovid_bert/clinicalcovid_bert_base_cased.ckpt.meta) • [`tensorflow.index`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/clinicalcovid_bert/clinicalcovid_bert_base_cased.ckpt.index) • [`pytorch_model.bin`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/clinicalcovid_bert/clinicalcovid_bert_base_cased.bin) • [`vocab.txt`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/clinicalcovid_bert/vocab.txt)
| `biocovid_bert_large_cased` | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/biocovid_bert/bert_config_bio_58k_large.json) • [`tensorflow weights`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/biocovid_bert/biocovid_bert_large_cased.ckpt.data-00000-of-00001) • [`tensorflow.meta`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/biocovid_bert/biocovid_bert_large_cased.ckpt.meta) • [`tensorflow.index`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/biocovid_bert/biocovid_bert_large_cased.ckpt.index) • [`pytorch_model.bin`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/biocovid_bert/biocovid_bert.bin) • [`vocab.txt`](https://s3.amazonaws.com/models.huggingface.co/bert/mananeau/biocovid_bert/vocab_cased_pubmed_pmc_30k.txt)

## Usage 
- Transformers library
- how to produce embeddings 


## Model and training description

### ClinicalCovidBERT 
- BERT base default configuration
- Cased 
- Initialized from [Bio+Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT)
- Using English `bert_base_cased` default vocabulary
- Using whole-word masking
- Pretrained on a preprocessed version of the CORD-19 dataset including titles, abstract and body text (approx. 1.5GB)
- The model was trained for X hours on one TPU v3-8.
- Tensorboard link
- Training parameters:
  - `train_batch_size`: 512
  - `max_seq_length`: 128
  - `max_predictions_per_seq`: 20
  - `num_train_steps`: 150000 
  - `num_warmup_steps`: 10000
  - `learning_rate`: 2e-5

### BioCovidBERT
- BERT large default configuration
- Cased 
- Initialized from [BioBERT-Large v1.1 (+ PubMed 1M)](https://github.com/dmis-lab/biobert) using their custom 30k vocabulary
- Using whole-word masking
- Pretrained on the same preprocessed version of the CORD-19 dataset including titles, abstract and body text (approx. 1.5GB)
- The model was trained for X hours on one TPU v3-8
- Tensorboard link
- Training parameters:
  - `train_batch_size`: 512
  - `max_seq_length`: 128
  - `max_predictions_per_seq`: 20
  - `num_train_steps`: 200000 
  - `num_warmup_steps`: 10000
  - `learning_rate`: 2e-5
  


## Train your own model 

### Data preprocessing

**TO DO**: The preprocessing script in the `preprocessing` folder. To download the data and preprocess it, use the `preprocessing.sh` script by running in the Terminal:
- `chmod +x clash_covid/preprocessing/preprocessing.sh`
- `./clash_covid/preprocessing/preprocessing.sh` 

### Training instructions

**TO DO: **

The TensorFlow code from the [official BERT repository](https://github.com/google-research/bert) was used for training.

## References

Emily Alsentzer, John Murphy, William Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew McDermott. 2019. Publicly available clinical BERT embeddings. In Proceedings of the 2nd Clinical Natural Language Processing Workshop, pages 72-78, Minneapolis, Minnesota, USA. Association for Computational Linguistics.

Lee, Jinhyuk, et al. "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." Bioinformatics 36.4 (2020): 1234-1240.
BioBERT

Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

## Acknowledgements

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)
