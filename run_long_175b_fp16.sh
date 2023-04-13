unset NCCL_DEBUG
GPUS_PER_NODE=8
MASTER_ADDR=10.174.136.212
MASTER_PORT=6001
NNODES=8
val=`expr $PADDLE_TRAINER_ID - 8`
NODE_RANK=$val
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $PADDLE_TRAINER_ID -le '7' ]
then 
    echo "id grather then 7, exit "
    exit
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

rm checkpoints/ -rf
CHECKPOINT_PATH=checkpoints/gpt2_175b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=./my-gpt2_text/my-gpt2-enwiki_text_document
#rm ${DATA_PATH}_*indexmap*

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

GPT_ARGS="--num-layers 96 \
          --hidden-size 12288 \
          --attention-dropout 0.1 \
          --hidden-dropout 0.1 \
          --num-attention-heads 96 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 1 \
          --global-batch-size 64 \
          --weight-decay 0.1  \
          --adam-beta1=0.9  \
          --adam-beta2=0.95  \
          --lr 0.00006 \
          --min-lr 0.000006 \
          --init-method-std 0.006 \
          --lr-decay-iters 90000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction 0.01 \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --train-iters 100000 \
          --split 949,50,1 \
          --recompute-granularity  \
          --recompute-method uniform \
          --fp16 \
          "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000000 \
             --eval-interval 100000 \
             --eval-iters 10"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 8 \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       2>&1 |tee temp.log
