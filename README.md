# Dialogue-Probe-Tasks-Public

The modified dataset containing the probe tasks and the script to generate the probe tasks can be found in `/Dataset`

## Training the model

Use the `--model` argument to train the appropriate model. The options include `hred, seq2seq, seq2seq_attn, transformer and bilstm_attn`

```
python Train.py --model bilstm_attn
```

## Evaluating on the ProbeTasks

The probe tasks are evaluated in `ProbeTasks.py`. The choice of selecting the probetasks can be done by selecting the appropriate column index in probe tasks. `ProbeTasks.py` currently evaluates models on the probetasks sequetially.

```
python ProbeTasks.py
```

## Plotting the results

The 
