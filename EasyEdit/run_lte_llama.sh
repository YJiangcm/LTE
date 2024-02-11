#!/bin/bash

for a in 'portability_r' 'portability_s' 'portability_l' 'locality_rs'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=EasyEdit/hparams/IKE/llama-7b \
            --data_dir=EasyEdit/data/knowedit/ZsRE/ZsRE-test-all.json \
            --eval_aspect=$a \
            --fluency
    done


for a in 'portability_r' 'portability_s' 'portability_l' 'locality_rs' 'locality_f'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=EasyEdit/hparams/IKE/llama-7b \
            --data_dir=EasyEdit/data/knowedit/wiki_counterfact/test_cf.json \
            --eval_aspect=$a \
            --fluency
    done


for a in 'portability_r' 'portability_s' 'portability_l' 'locality_rs' 'locality_f'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=EasyEdit/hparams/IKE/llama-7b \
            --data_dir=EasyEdit/data/knowedit/wiki_recent/recent_test.json \
            --eval_aspect=$a \
            --fluency
    done

for a in 'locality_rs'
    do
        python run_knowedit.py \
            --editing_method=IKE \
            --hparams_dir=EasyEdit/hparams/IKE/llama-7b \
            --data_dir=EasyEdit/data/knowedit/WikiBio/wikibio-test-all.json \
            --eval_aspect=$a \
            --fluency
    done