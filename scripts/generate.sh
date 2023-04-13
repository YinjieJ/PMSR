max_tokens=14000
UPDATE_FREQ=1
pointer_layer=-2

sp=atom
m=pistachio

for task in 'uspto_mit' 
do
data_dir=data/$task/$sp
save_dir=finetune/$task'_'$m'_'$sp

# rm -rf $save_dir/generate
mkdir -p $save_dir/generate

CUDA_VISIBLE_DEVICES=$1 fairseq-generate $data_dir \
    --user-dir src --truncate-source --source-lang tgt --target-lang src \
    --task translation --beam 20 --nbest 15 \
    --gen-subset 'test' \
    --path $save_dir/model/checkpoint_last.pt \
    --results-path $save_dir/generate \
    --max-tokens "$max_tokens" --required-batch-size-multiple 1 \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 5 --seed 42
done