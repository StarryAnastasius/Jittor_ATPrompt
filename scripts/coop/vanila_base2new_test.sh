#!/bin/bash

# custom config
DATA=/root/prompt_dataset
CFG=vit_b16
SHOTS=16
CSC=False
CTP=end

TRAINER=CoOp
EPO=100  # 训练时的最大epoch（用于目录生成）
NCTX=4
DATASET=$1  # caltech101 等数据集

# 要评测的epoch列表
EVAL_EPOCHS=(1 2 3 4 5)

for SEED in 1 #2 3
do
    for EVAL_EPOCH in "${EVAL_EPOCHS[@]}"
    do
        # 模型保存目录（不变）
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}
        # 评测结果输出目录（可加上当前评测的epoch，避免覆盖）
        OUT_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}/evaluation/epoch${EVAL_EPOCH}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoOp/${CFG}.yaml \
                --model-dir ${DIR} \
                --output-dir ${OUT_DIR} \
                --load-epoch ${EVAL_EPOCH} \
                --eval-only \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base \
                OPTIM.MAX_EPOCH ${EPO}
    done
done