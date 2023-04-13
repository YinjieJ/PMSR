TASK=uspto_mit
SP=atom

data_dir=raw_data/$TASK/$SP
main_dir=data/$TASK/$SP

rm -rf $main_dir
mkdir -p $main_dir

fairseq-preprocess \
  --source-lang src \
  --target-lang tgt \
  --trainpref $data_dir/train \
  --validpref $data_dir/valid \
  --testpref $data_dir/test \
  --destdir $main_dir \
  --workers 10 \
  --joined-dictionary \
  --srcdict pre_data/dict.src.txt

# fairseq-preprocess \
#   --trainpref $data_dir/train.slabel \
#   --validpref $data_dir/valid.slabel \
#   --testpref $data_dir/test.slabel \
#   --destdir $main_dir/label \
#   --workers 10 \
#   --only-source
