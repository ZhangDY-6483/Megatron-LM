GPUS_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=checkpoints/gpt2_1.3b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=./my-gpt2_text/my-gpt2-enwiki_text_document
rm ${DATA_PATH}_*indexmap*
rm checkpoints/ -rf

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

GPT_ARGS="--num-layers 24 \
          --hidden-size 2048 \
          --attention-dropout 0.0 \
          --hidden-dropout 0.0 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 16 \
          --global-batch-size 32 \
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
          --recompute-granularity full \
          --recompute-method uniform \
          "
          #--bf16 \
          #--use-flash-attn \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 100000 \
             --eval-interval 100000 \
             --eval-iters 10"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       2>&1 |tee 5_bf16_1.3b_222_nofla.log
