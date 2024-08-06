#!/bin/bash

for s in 10 100 500 1000
do
    for a in 'portability_r' 'portability_s' 'portability_l' 'locality_rs'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=hparams/IKE/llama-7b \
            --data_dir=data/knowedit/ZsRE/ZsRE-test-all.json \
            --ds_size=$s \
            --eval_aspect=$a
    done
done


for s in 10 100 500 1000
do
    for a in 'portability_r' 'portability_s' 'portability_l' 'locality_rs' 'locality_f'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=hparams/IKE/llama-7b \
            --data_dir=data/knowedit/wiki_counterfact/test_cf.json \
            --ds_size=$s \
            --eval_aspect=$a
    done
done


for s in 10 100 500 1000
do
    for a in 'portability_r' 'portability_s' 'portability_l' 'locality_rs' 'locality_f'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=hparams/IKE/llama-7b \
            --data_dir=data/knowedit/wiki_recent/recent_test.json \
            --ds_size=$s \
            --eval_aspect=$a
    done
done


for s in 10 100 500 1000
do
    for a in 'locality_rs'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=EasyEdit/hparams/IKE/llama-7b \
            --data_dir=data/knowedit/WikiBio/wikibio-test-all.json \
            --ds_size=$s \
            --eval_aspect=$a
    done
done