# English-Spanish Translator

To run, you first need to [download the dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset) as a CSV named "data.csv" and place it in a new folder called "data".

To train a new model, call train.py. You can change the hyperparameters of the model by changing the variables at the top of the file.

```bash
python train.py
```

Once the model is fully trained, you can interact with it in a REPL with interactive.py. Make sure to change the path to the checkpoint file with the variable model_load_path in interactive.py.

```bash
python interactive.py
```

To create the grpahs present in our report and slideshow, use graphs.py. Pass as arguments both the path to the log.txt file created during training and either "bleu" or "loss" depending on what you want to display in the graph.

```bash
python graphs.py PATH [bleu, loss]
```

## References
- [Andrej Karpathy's video on GPTs](https://youtu.be/kCc8FmEb1nY?si=n_1X2I4VNOj8Y3Lk): gives a good overview of GPTs, and showcases some useful functions in PyTorch. Some of our decoder code is based on code in this video.
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- [English Spanish translation dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset): Kaggle dataset for English to Spanish translation.
- [A good overview of BLEU score](https://www.geeksforgeeks.org/nlp/nlp-bleu-score-for-evaluating-neural-machine-translation-python/)
- [GPT-3 paper has a lot of good implementation information, like cosine lr scheduler with warmup](https://arxiv.org/pdf/2005.14165)