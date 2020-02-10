#!/bin/bash

python3 prepareData.py go_def_in_obo.tsv go_names.txt go_defs.txt

python3 extract_features.py --input_file go_defs.txt --output_file layers.json --bert_model fine_tune_lm_bioBERT --layers -2 --no_cuda

python3 readjson.py go_names.txt layers.json bertAsServiceVecs.csv

rm layers.json
rm go_names.txt
rm go_defs.txt

