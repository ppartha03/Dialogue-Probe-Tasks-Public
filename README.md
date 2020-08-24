# Dialogue-Probe-Tasks-Public
We evaluate the dialogue generation models on their performance on probe tasks as discussed in the paper here. We observe the dialogue models lack in understanding the context better. More so we observe that the transformer model's large representation manifold affect the model's ability to learn underlying structures. This was evident in their performance on the probe-tasks. We discuss the results in detail in the paper [].

The modified dataset containing the probe tasks and the script to generate the probe tasks can be found in `/Dataset`

### Package Requirements
```
python==3.6
torch==1.2
torchtext
seaborn
bert-embedding
pandas
scikit-learn
```

## Training the model

Use the `--model` argument to train the appropriate model. The options include `hred, seq2seq, seq2seq_attn, transformer and bilstm_attn`

```
python Train.py --model bilstm_attn --dataset MultiWoZ --batch_size 32 --s2s_hidden_size 256 --s2s_embedding_size 128
```

## Evaluating on the ProbeTasks

The probe tasks are evaluated in `ProbeTasks.py`. The choice of selecting the probetasks can be done by selecting the appropriate column index in probe tasks. `ProbeTasks.py` currently evaluates models on the probetasks sequetially.

```
python ProbeTasks.py
```
The command writes the results on to a csv that can be used to plot the graphs in the appendix. The tables can be obtained from the CSV.

## Plotting the results

To generate the graphs use the following command from `Utils/`:

```
python ProbeTasks.py --dataset MultiWoZ
```
Also you can use `Utils/format_to_table.py` to generate latex table with error range.

```
python format_to_table.py --dataset MultiWoZ
```
## Alternate Metrics

To select models with alternate metrics run `Utils/Alternate_Metrics.py`. The code will return a csv with the epoch number for the best model with METEOR, ROUGE-F1, BERT (Average), BLEU.

```
python Alternate_Metrics.py --dataset MultiWoZ
```

To precompute the BERT embeddings for the vocabulary, use `python bertembeddings.py --dataset MultiWoZ`.

### Citation:

If you find this work useful and use it in your own research, please consider citing our [paper](link).
```
@misc{parthasarathi2020dialogueprobetasks,
    title={},
    author={How To Evaluate Your Dialogue System: Probe Tasks as an Alternative for Token-level Evaluation Metrics},
    year={2020},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
