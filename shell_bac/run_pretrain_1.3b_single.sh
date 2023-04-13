CHECKPOINT_PATH=checkpoints/gpt2_1.3b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
# DATA_PATH=./wikitext_103_en/wikitext_103_en_text_document
DATA_PATH=./my-gpt2_text/my-gpt2_text_document
rm ${DATA_PATH}_*indexmap*
rm checkpoints/ -rf

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

export MASTER_ADDR=`hostname -i`
export MASTER_PORT=6000
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPT_ARGS="--num-layers 24 \
          --hidden-size 1024 \
          --attention-dropout 0.0 \
          --hidden-dropout 0.0 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 4 \
          --global-batch-size 4 \
          --lr 0.00015 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --train-iters 10000 \
          --bf16 \
          --checkpoint_activations \
          --use-flash-attn \
          "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 1000 \
             --eval-interval 10 \
             --eval-iters 10"

python pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
