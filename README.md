# GeneOntologyEncoders
Bert as A Service, Elmo and Precomputed Vector encoders

## Bert as A Service 
Given a list of go terms and go definitions, use the code in this folder to get GO encoders for each term with dim 768. It uses the same method as Bert as a service, where we take the 11th layer of the output of the bert transformer and average it over the sentence. Specifically to get a csv file of the final encoders for the GO terms, run GetVecFile.sh. The final file is saved as bertAsServiceVecs.csv with the first column being the name of the go term and the second term being the encoding of the go_term. 

To run: 

1) use a csv file with the same structure as go_def_in_obo.tsv and change go_def_in_obo.tsv in GetVecFile.sh to whatever your csv file is named

2) switch the name of bert model in GetVecFile.sh from fine_tune_lm_bioBERT to whatever folder your model is saved in.

## Elmo
If you want full length model with the 1024 dim, use elmo1024.sh. However if you want a model that is reduced to 768 to be the same dimension of BERT, use elmo768.sh

