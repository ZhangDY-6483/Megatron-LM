GPUS_PER_NODE=8
MASTER_ADDR=10.215.195.86
MASTER_PORT=6000
NNODES=8
NODE_RANK=$PADDLE_TRAINER_ID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=checkpoints/gpt2_175b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=./my-gpt2_text/my-gpt2-enwiki_text_document
#rm ${DATA_PATH}_*indexmap*
rm checkpoints/ -rf

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

GPT_ARGS="--num-layers 96 \
          --hidden-size 12288 \
          --attention-dropout 0.0 \
          --hidden-dropout 0.0 \
          --num-attention-heads 96 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 16 \
          --global-batch-size 16 \
          --lr 0.00015 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --train-iters 30 \
          --recompute-granularity full \
          --recompute-method uniform \
          --bf16 \
          --use-flash-attn \
          "
          #--bf16 \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000000 \
             --eval-interval 100000 \
             --eval-iters 1"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 8 \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       2>&1 |tee 30_175b_test_bf16_124.log
