<img src="./images/bertvision.png">

## Welcome to BERTVision

We present a highly parameter-efficient approach for a wide range of NLP tasks that significantly reduces the need for extended BERT fine-tuning. Our method uses information from the hidden state activations of each BERT transformer layer, which is discarded during typical BERT inference. Our best model achieves maximal BERT performance at a fraction of the training time and GPU/TPU expense. Performance is further improved by ensembling our model with BERT’s predictions. Furthermore, we find that near optimal performance can be achieved for some NLP tasks using less training data. 

This repository contains the code, models, and documentation for the evaluation of leveraging parameter-efficient models for their potential utility in performing the NLP tasks of span annotation, document classification, grammatical validation, sentiment, paraphrase detection, sentence similarity, and more.

All models were trained on the hidden embedding activation states of BERT-base and BERT-large uncased and evaluated on the [Stanford Question Answering Dataset 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (aka [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)) and the [General Language Understanding Evaluation](https://gluebenchmark.com/) ([GLUE](https://gluebenchmark.com/)) benchmark data.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/cbenge509/BERTVision/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
