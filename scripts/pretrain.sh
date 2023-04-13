total_updates=500000000000
warmup_updates=4000      
# lr=3e-05
lr=2
max_tokens=18000
UPDATE_FREQ=1
pointer_layer=-2
data_dir=pre_data/atom
task=pistachio
save_dir=result/$task

# rm -rf $save_dir
mkdir -p $save_dir

cp $0 $save_dir

fairseq-train $data_dir \
    --user-dir src --truncate-source --source-lang src --target-lang tgt \
    --task auto_encoding_regressive --arch pmsr_base --criterion label_smoothed_cross_entropy_with_masked_lm --label-smoothing 0.1 \
    --share-all-embeddings --share-decoder-input-output-embed \
    --optimizer adam --adam-betas "(0.9, 0.998)" --adam-eps 1e-08 --max-epoch 100\
    --lr-scheduler noam --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --dropout 0.1 --attention-dropout 0.1 --clip-norm 0.5 \
    --skip-invalid-size-inputs-valid-test --validate-interval 1 --max-tokens-valid 14000 \
    --max-tokens "$max_tokens" --required-batch-size-multiple 1 \
    --num-workers 5 --seed 42\
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --encoder-layers 4 \
    --encoder-embed-dim 768 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 \
    --decoder-layers 4 \
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
    --keep-last-epochs 100 \
    --keep-best-checkpoints -1 \
    --best-checkpoint-metric loss \
    --log-format json \
    --save-dir $save_dir/model --tensorboard-logdir $save_dir/log \
    --ddp-backend=legacy_ddp \
    --find-unused-parameters
    # --reset-optimizer --reset-dataloader --reset-meters \
    # --wandb-project PMSR \
    # --fp16 \
    # --find-unused-parameters \
    # --no-epoch-checkpoints \ 