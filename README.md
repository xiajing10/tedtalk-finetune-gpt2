# Note
Code base was developed in 2020, that's why gpt-2 here :)

Finetuning code will be uploaded later.

Please pay attention to dependencies' version and updates may be needed.

# Overview
Finetune gpt2 model with tedtalk transcript and metadata (e.g. keywords, occupation, etc) as keywords for topic control.

Raw data is from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/ted-talks/data)

# Dependencies
transformers
pytorch

# Notebooks
## Fintuning code
[Google Colab: transformer-gpt2-finetuning](https://colab.research.google.com/drive/1z70k27dBYTkNDaSZKuKRYAVfJUQ1S5Ex?usp=sharing)
It is highly recommended to run on GPU.

## Generation code
[Google Colab: transformer-gpt2-finetuning](https://colab.research.google.com/drive/1oDGbnPVwanWK5ssXwRwehBA2f9t4K9gN?usp=sharing)
Generation part can work withou GPU, but longer processing time shall be expected.

