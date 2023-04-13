total_updates=500000000000
total_epochs=2000
warmup_updates=100
# lr=3e-05
lr=1e-3
max_tokens=14000
UPDATE_FREQ=1
pointer_layer=-2

m=pistachio
sp=atom
# task=USPTO50K
# task=USPTO_full
# task=denoise
task=uspto_mit

data_dir=data/$task/$sp
save_dir=finetune/$task'_'$m'_'$sp

# rm -rf $save_dir
mkdir -p $save_dir

cp $0 $save_dir

CUDA_VISIBLE_DEVICES=$1 fairseq-train $data_dir \
    --user-dir src --truncate-source --source-lang tgt --target-lang src \
    --task translation --arch pmsr_base --criterion label_smoothed_cross_entropy --label-smoothing 0.0 \
    --share-all-embeddings --share-decoder-input-output-embed \
    --optimizer adam --adam-betas "(0.9, 0.998)" --adam-eps 1e-08 --max-epoch $total_epochs\
    --lr-scheduler noam --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --dropout 0.1 --attention-dropout 0.1 --clip-norm 1 \
    --skip-invalid-size-inputs-valid-test --validate-interval 1 --max-tokens-valid 14000 \
    --max-tokens "$max_tokens" --required-batch-size-multiple 1 \
    --num-workers 20 --seed 42\
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --encoder-layers 6 \
    --encoder-embed-dim 768 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 \
    --decoder-layers 6 \
    --decoder-embed-dim 768 \
    --decoder-ffn-embed-dim 2048 \
    --decoder-attention-heads 8 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --source-position-markers 0 --update-freq "$UPDATE_FREQ" \
    --train-subset train \
    --valid-subset valid \
    --save-interval-updates 10000 \
    --keep-interval-updates 10 \
    --keep-interval-updates-pattern 100000 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints -1 \
    --best-checkpoint-metric accuracy \
    --log-format json \
    --save-dir $save_dir/model --tensorboard-logdir $save_dir/log \
    --ddp-backend=legacy_ddp \
    --find-unused-parameters \
    --report-accuracy \
    --maximize-best-checkpoint-metric \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file result/$m/model/checkpoint_last.pt
    # --wandb-project PMSR \
    # --fp16 \
    # --find-unused-parameters \
    # --no-epoch-checkpoints \ 