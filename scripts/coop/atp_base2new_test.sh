# #!/bin/bash

# # custom config
# DATA="D:/Downloads/ATPrompt-main/data"
# CFG=rn50_ep50
# SHOTS=16
# CSC=False
# CTP=end

# TRAINER=CoOp_ATP
# EPO=50
# NCTX=2
# DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

# for SEED in 1 #2 3 4 5
# do
#         DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp_${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}

#         CUDA_VISIBLE_DEVICES=0 /mnt/d/anaconda3/envs/atprompt/python.exe ../../train.py \
#                 --root ${DATA} \
#                 --seed ${SEED} \
#                 --trainer ${TRAINER} \
#                 --dataset-config-file ../../configs/datasets/${DATASET}.yaml \
#                 --config-file ../../configs/trainers/CoOp/${CFG}.yaml \
#                 --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp_${CTP}/${DATASET}/seed${SEED} \
#                 --model-dir ${DIR} \
#                 --load-epoch ${EPO} \
#                 --eval-only \
#                 DATASET.SUBSAMPLE_CLASSES new \
#                 TRAINER.COOP.N_CTX ${NCTX} \
#                 TRAINER.COOP.CSC ${CSC} \
#                 TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
#                 TRAINER.ATPROMPT.USE_ATPROMPT True \
#                 TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
#                 TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
#                 TRAINER.ATPROMPT.N_ATT3 ${NCTX}
# done

#!/bin/bash

# custom config
DATA=/root/prompt_dataset
CFG=vit_b16
SHOTS=16
CSC=False
CTP=end

TRAINER=CoOp_ATP
EPO=10 #要改的
NCTX=2
DATASET=$1


EVAL_EPOCHS=(1 2 3 4 5 6 7 8 9 10)

for SEED in 2 3
do
    for EVAL_EPOCH in "${EVAL_EPOCHS[@]}"
    do

        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}
        OUT_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}/evaluation/epoch${EVAL_EPOCH}
        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoOp/${CFG}.yaml \
                --output-dir ${OUT_DIR} \
                --model-dir ${DIR} \
                --load-epoch ${EVAL_EPOCH} \
                --eval-only \
                DATASET.SUBSAMPLE_CLASSES new \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT3 ${NCTX}
    done
done