mkdir /local/auppunda/elmo_2017_1024
mkdir /local/auppunda/elmo_2017_1024_1
#UDA_VISIBLE_DEVICES=5 /local/auppunda/anaconda3/bin/python3 ElmoC/encoder/do_model.py --fix_word_emb --vocab_list /local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt --w2v_emb /local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle --lr 0.0075  --qnli_dir /local/auppunda/goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls --batch_size_label 24 --result_folder /local/auppunda/elmo_2017_1024 --epoch 18  --use_cuda --metric_option cosine --def_emb_dim 768 --bilstm_dim 1024 --main_dir /local/auppunda/goAndGeneAnnotationMar2017 --write_vector  --label_desc_dir /local/auppunda/deepgo/data/go_def_in_obo.csv  --average_layers  > test
#CUDA_VISIBLE_DEVICES=5 /local/auppunda/anaconda3/bin/python3 ElmoC/encoder/do_model.py --fix_word_emb --vocab_list /local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt --w2v_emb /local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle --lr 0.007  --qnli_dir /local/auppunda/goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls --batch_size_label 24 --result_folder /local/auppunda/elmo_2017_1024_1 --epoch 3  --use_cuda --metric_option cosine --def_emb_dim 768 --bilstm_dim 1024 --main_dir /local/auppunda/deepgo/goAndGeneAnnotationMar2017 --write_vector  --label_desc_dir /local/auppunda/deepgo/data/go_def_in_obo.csv --model_load /local/auppunda/elmo_2017_1024/best_state_dict.pytorch --average_layers  > test1
#CUDA_VISIBLE_DEVICES=1 /local/auppunda/anaconda3/bin/python3 ElmoC/encoder/do_model.py --fix_word_emb --vocab_list /local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt --w2v_emb /local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle --lr 0.0065  --qnli_dir /local/auppunda/auppunda/goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls --batch_size_label 8 --result_folder /local/auppunda/elmo_20171 --epoch 5  --use_cuda --metric_option cosine --def_emb_dim 768 --bilstm_dim 1024 --main_dir /local/auppunda/auppunda/goAndGeneAnnotationMar2017 --write_vector  --label_desc_dir /local/auppunda/auppunda/deepgo/data/go_def_in_obo.csv --model_load /local/auppunda/auppunda/elmo_2017/best_state_dict.pytorch --average_layers --reduce_cls_vec > test


server='/local/auppunda'

## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
#bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
#mkdir $bert_output_dir

result_folder=$bert_output_dir/$metric_option'.768.ElmoVec' #$def_emb_dim.'clsVec'
mkdir $result_folder

model_load=$server/'elmo_2017_1024/best_state_dict.pytorch'

#pair='Yeast' # 'HumanMouse'
#outDir=$server/$pair'PPI3ontology/qnliFormatData17'
#mkdir $outDir

#finalDir=$outDir/$metric_option'.768.reduce300ClsVec'
#mkdir $finalDir
w2v_emb='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt'

#for point in {11400..11700..300} # 12600 11700
#do

#echo ' '
#echo 'iter '$point

#savePickle=$outDir/'GeneDict2test.'$point'.pickle'

#saveDf=$outDir/'PPI2testDef.'$point'.txt'
#test_file=$saveDf

#write_score=$finalDir/'PPI2testDef.'$point'.score.txt'

#CUDA_VISIBLE_DEVICES=5 /local/auppunda/anaconda3/bin/python ElmoC/encoder/do_model.py --average_layers --vocab_list $vocab_list --w2v_emb $w2v_emb --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 156  --pregenerated_data $pregenerated_data --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --model_load $model_load --write_score $write_score --test_file $test_file
## set epoch=0 for testing
#CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 64 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file > $result_folder/test1.log

#paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

# rm -f $test_file
#rm -f $write_score

#done

pair='HumanMouse' # 'HumanMouse'
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'
mkdir $outDir
finalDir=$outDir/$metric_option'.768.ElmoVec'
mkdir $finalDir

#w2v_emb='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
#ocab_list='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt'
#w2v_emb='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
#vocab_list='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt' 

for point in {300..300..300} # 12600
do

echo ' '
echo 'iter '$point

savePickle=$outDir/'GeneDict2test.'$point'.pickle'

saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
test_file=$saveDf
write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'

CUDA_VISIBLE_DEVICES=7 /local/auppunda/anaconda3/bin/python ElmoC/encoder/do_model.py --average_layers --vocab_list $vocab_list --w2v_emb $w2v_emb --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 128  --pregenerated_data $pregenerated_data --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim  --model_load $model_load --write_score $write_score --test_file $test_file

## set epoch=0 for testing
#CUDA_VISIBLE_DEVICES=5 /local/auppunda/auppunda/anaconda3/bin/python BERT/encoder/do_model.py --main_dir $work_dir --average_layer --qnli_dir $data_dir --batch_size_label 64 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file > $result_folder/test1.log

paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

# rm -f $test_file
#rm -f $write_score

done


pair='FlyWorm' # 'HumanMouse'
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'
mkdir $outDir
finalDir=$outDir/$metric_option'.768.ElmoVec'
mkdir $finalDir

#w2v_emb='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
#vocab_list='/local/auppunda/goAndGeneAnnotationMar2017/word_pubmed_intersect_GOdb.txt'

for point in {600..600..300} # 12600
do

echo ' '
echo 'iter '$point

savePickle=$outDir/'GeneDict2test.'$point'.pickle'

saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
test_file=$saveDf
write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'

CUDA_VISIBLE_DEVICES=7 /local/auppunda/anaconda3/bin/python ElmoC/encoder/do_model.py --average_layers --vocab_list $vocab_list --w2v_emb $w2v_emb --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 64  --pregenerated_data $pregenerated_data --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim  --model_load $model_load --write_score $write_score --test_file $test_file

## set epoch=0 for testing
#CUDA_VISIBLE_DEVICES=5 /local/auppunda/auppunda/anaconda3/bin/python BERT/encoder/do_model.py --main_dir $work_dir --average_layer --qnli_dir $data_dir --batch_size_label 64 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file > $result_folder/test1.log

paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

 #rm -f $test_file
 #rm -f $write_score

done

