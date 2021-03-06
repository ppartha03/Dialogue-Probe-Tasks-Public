# Dialogue-Probe-Tasks-Public
We evaluate the dialogue generation models on their performance on probe tasks as discussed in our [paper](https://arxiv.org/abs/2008.10427). The codebase shares the dataset used in the paper and the codebase that can be used to recreate the results in the paper. We hope the codebase serves as a base to work on further improvements on the results of the paper.

The modified dataset containing the probe tasks and the script to generate the probe tasks can be found in `/dataset`

### Package Requirements
We use python 3.6 for all the experiments in the paper.
```
pip install -r requirements.txt
```

## Training the model

Use the `--model` argument to train the appropriate model. The options include `hred, seq2seq, seq2seq_attn, transformer and bilstm_attn`

```
python train.py --model bilstm_attn --dataset MultiWoZ --batch_size 32 --s2s_hidden_size 256 --s2s_embedding_size 128
```

## Evaluating on the ProbeTasks

The probe tasks are evaluated in `probetasks.py`. The choice of selecting the probetasks can be done by selecting the appropriate column index in probe tasks. `probetasks.py` currently evaluates models on the probetasks sequetially.

```
python probetasks.py
```
The command writes the results on to a csv that can be used to plot the graphs in the appendix. The tables can be obtained from the CSV.

## Plotting the results

To generate the graphs use the following command from `utils/`:

```
python ProbeTasks.py --dataset MultiWoZ
```
Also you can use `utils/format_to_table.py` to generate latex table with error range.

```
python format_to_table.py --dataset MultiWoZ
```
## Alternate Metrics

To select models with alternate metrics run `utils/alternate_metrics.py`. The code will return a csv with the epoch number for the best model with METEOR, ROUGE-F1, BERT (Average) and BLEU.

```
python alternate_metrics.py --dataset MultiWoZ
```

To precompute the BERT embeddings for the vocabulary, use `python bert_embeddings.py --dataset MultiWoZ`.

### Citation:

If you find this work useful and use it in your own research, please consider citing our [paper](https://arxiv.org/abs/2008.10427).
```
  @inproceedings{
  parthasarathi2021Encoder,
  author       = {Parthasarathi, Prasanna and Chandar, Sarath and Pineau, Joelle},
  title        = {Do Encoder Representations of Generative Dialogue Models Encode Sufficient Information about the Task ?},
  year         = {2021},
  booktitle    = {Proceedings of the 22nd Annual SIGdial Meeting on Discourse and Dialogue},
  publisher    = {Association for Computational Linguistics},
}
```
