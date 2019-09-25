# longLM
Using pretrained LMs for long doc classification.

The only new python library you’ll need can be gotten with:

`pip install pytorch_pretrained_bert`

# Running BERT
training and evaluation is ran with:

`python run_classifier.py --task_name arxiv --do_train --do_eval --data_dir ./data/arxiv/ --bert_model bert-base-uncased  --do_lower_case --max_seq_length 256 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./arxiv1`

Datasets are expected to be in TSV format. If you want to add a new dataset, you’ll have to copy paste some code inside the dataset utils file, but it’s pretty easy to do.

# Running Long Bert

`python bidirectional_run_classifier.py --task_name arxiv --do_train  --data_dir ./arxiv/data/arxiv/ --do_lower_case --bert_model bert-base-uncased  --max_seq_length 256 --learning_rate 2e-5 --num_train_epochs 3.0 --train_batch_size 128 --eval_batch_size 128 --seq_segments 16 --experiment attention --output_dir ./arxiv1`

The experiment argument has 4 options: attention, base, long, ablation.

* base: original bert
* long: uses an lstm to keep track of all bert hidden representations, but backprop over the first
* attention: uses an lstm + attention mechanism to backprop over more than the first representation
* ablation: concat all the hidden representations
