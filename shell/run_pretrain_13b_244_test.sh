GPUS_PER_NODE=8
MASTER_ADDR=10.67.182.140
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=checkpoints/gpt2_13b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=./my-gpt2_text/my-gpt2-enwiki_text_document
#rm ${DATA_PATH}_*indexmap*
rm checkpoints/ -rf

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

GPT_ARGS="--num-layers 40 \
          --hidden-size 5120 \
          --attention-dropout 0.0 \
          --hidden-dropout 0.0 \
          --num-attention-heads 40 \
          --seq-length 4096 \
          --max-position-embeddings 4096 \
          --micro-batch-size 4 \
          --global-batch-size 16 \
          --lr 0.00015 \
          --min-lr 1.0e-5 \
          --weight-decay 1e-2 \
          --clip-grad 1.0 \
          --lr-warmup-fraction .001 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --data-impl mmap \
          --split 949,50,1 \
          --distributed-backend nccl \
          --train-iters 10000 \
          --bf16 \
          --recompute-granularity full \
          --recompute-method uniform \
          --use-flash-attn \
          "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 100000 \
             --eval-interval 100000 \
             --eval-iters 10"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       2>&1 |tee 13b_244_6.log
