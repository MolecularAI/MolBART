WORLD_SIZE=8
MP_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=checkpoints/bert_345m
VOCAB_FILE=bart_vocab.txt
DATA_PATH=my-bert_text_sentence
BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --lr-decay-iters 990000 \
           --train-iters 2000000 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
	   --micro-batch-size 4 \	   
           --global-batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_bert.py \
                $BERT_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data-path $DATA_PATH \
                --tensor-model-parallel-size $MP_SIZE \
                --DDP-impl torch