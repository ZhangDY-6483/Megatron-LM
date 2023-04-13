GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST=8.0
if [ $PADDLE_TRAINER_ID -gt '7' ]
then 
    echo "id grather then 7, exit "
    exit
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"

rm checkpoints/ -rf
CHECKPOINT_PATH=checkpoints/gpt2_175b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=./my-gpt2_text/my-gpt2-enwiki_text_document
#rm ${DATA_PATH}_*indexmap*

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

GPT_ARGS="--num-layers 4 \
          --seed 1234 \
          --hidden-size 12288 \
          --attention-dropout 0.0 \
          --hidden-dropout 0.0 \
          --num-attention-heads 96 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 1 \
          --global-batch-size 1 \
          --lr 0.00005 \
          --min-lr 0.00001 \
          --lr-decay-iters 360000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --train-iters 10000 \
          --use-memory-attn \
          --recompute-granularity selective \
          "
          #--bf16 \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000000 \
             --eval-interval 100000 \
             --eval-iters 1"

export NVIDIA_TF32_OVERRIDE=0

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ./pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 2 \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       2>&1 |tee temp.log
