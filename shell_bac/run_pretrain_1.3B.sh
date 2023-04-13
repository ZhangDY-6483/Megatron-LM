GPUS_PER_NODE=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#DATAPATH=./wikitext_103_en/wikitext_103_en_text_document
DATA_PATH=./my-gpt2_text/my-gpt2_text_document
CHECKPOINT_PATH=gpt_model/gpt2_1.3b
LOAD_PATH=gpt_model/gpt2_1.3b

source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

        #--tensor-model-parallel-size 2 \
        #--pipeline-model-parallel-size 2 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --recompute-granularity full \
        --recompute-method uniform \
        --num-layers 24 \
        --hidden-size 2048 \
        --tensorboard-dir ./tflog \
        --num-attention-heads 16 \
        --use-flash-attn \
        --micro-batch-size 4 \
        --global-batch-size 8 \
        --seq-length 1024 \
        --dataloader-type cyclic \
        --max-position-embeddings 1024 \
        --train-iters 10 \
        --lr-decay-iters 160000 \
        --save $CHECKPOINT_PATH \
        --load $LOAD_PATH \
        --data-path $DATAPATH \
        --vocab-file gpt2-vocab.json \
        --merge-file gpt2-merges.txt \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.0001 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .001 \
        --log-interval 1 \
        --save-interval 0 \
        --num-workers 1 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --bf16
