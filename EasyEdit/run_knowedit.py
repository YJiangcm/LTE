import os.path
import sys
import json
import random
sys.path.append('..')
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset, ZsreDataset

import argparse
# import nltk
# nltk.download('punkt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--eval_aspect',default=None, type=str)
    parser.add_argument('--fluency', default=False, action='store_true')

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    else:
        raise NotImplementedError
    

    datas = KnowEditDataset(args.data_dir)
    if args.ds_size is not None:
        datas = random.sample(datas, args.ds_size)
    args.data_dir = args.data_dir.split('/')[-2]

    prompts=[]
    subjects=[]
    target_new = []
    aspect_prompts=[]
    aspect_ans=[]
    for i in range(len(datas)):
        if datas[i][args.eval_aspect]:
            if 'zsre' in args.data_dir.lower() or 'wikibio' in args.data_dir.lower():
                if datas[i][args.eval_aspect][0]["prompt"] and datas[i][args.eval_aspect][0]["ground_truth"]:
                    prompts.append(datas[i]['prompt'])
                    subjects.append(datas[i]['subject'])
                    target_new.append(datas[i]['target_new'])
                    if args.eval_aspect == 'locality_rs':
                        aspect_prompts.append(datas[i][args.eval_aspect][0]["prompt"])
                        aspect_ans.append(datas[i][args.eval_aspect][0]["ground_truth"][0])
                    else:
                        aspect_prompts.append(datas[i][args.eval_aspect][0]["prompt"])
                        aspect_ans.append(datas[i][args.eval_aspect][0]["ground_truth"])
            else:
                if datas[i][args.eval_aspect][0]["prompt"] and len(datas[i][args.eval_aspect][0]["ground_truth"][0]) > 0:
                    prompts.append(datas[i]['prompt'])
                    subjects.append(datas[i]['subject'])
                    target_new.append(datas[i]['target_new'])
                    aspect_prompts.append(datas[i][args.eval_aspect][0]["prompt"])
                    aspect_ans.append(datas[i][args.eval_aspect][0]["ground_truth"][0][0])
        
    if args.eval_aspect == 'locality_rs':
        locality_inputs = {
            'Relation_Specificity':{
            'prompt': aspect_prompts,
            'ground_truth': aspect_ans
            },
        }
        portability_inputs = None

    elif args.eval_aspect == 'locality_f':
        locality_inputs = {
            'Forgetfulness':{
            'prompt': aspect_prompts,
            'ground_truth': aspect_ans
            },
        }
        portability_inputs = None

    elif args.eval_aspect == 'portability_r':
        portability_inputs = {
            'Reasoning':{
            'prompt': aspect_prompts,
            'ground_truth': aspect_ans
            },
        }
        locality_inputs = None

    elif args.eval_aspect == 'portability_s':
        portability_inputs = {
            'Subject_Aliasing':{
            'prompt': aspect_prompts,
            'ground_truth': aspect_ans
            },
        }
        locality_inputs = None

    elif args.eval_aspect == 'portability_l':
        portability_inputs = {
            'Logical_Generalization':{
            'prompt': aspect_prompts,
            'ground_truth': aspect_ans
            },
        }
        locality_inputs = None
    
    else:
        raise KeyError
    
    print(args.data_dir)
    print("Number of test examples: ", len(prompts))

    hparams = editing_hparams.from_hparams(args.hparams_dir)

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        train_ds=None,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True,
        test_generation=args.fluency
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.data_dir}_{args.eval_aspect}_results.json'), 'w'), indent=4)

    if args.editing_method == 'IKE':
        print("Edit Success: ", round(sum([i["post"]["rewrite_acc"] for i in metrics])/len(metrics) * 100, 2))
        if args.eval_aspect == "portability_r":
            print("Portability Reasoning: ", round(sum([i["post"]["portability"]["Reasoning_acc"] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "portability_s":
            print("Portability Subject_Aliasing: ", round(sum([i["post"]["portability"]["Subject_Aliasing_acc"] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "portability_l":
            print("Portability Logical_Generalization: ", round(sum([i["post"]["portability"]["Logical_Generalization_acc"] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "locality_rs":
            print("Locality Relation_Specificity: ", round(sum([i["post"]["locality"]["Relation_Specificity_acc"] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "locality_f":
            print("Locality Forgetfulness: ", round(sum([i["post"]["locality"]["Forgetfulness_acc"] for i in metrics])/len(metrics) * 100, 2))
        else:
            raise KeyError
    
    else:
        print("Edit Success: ", round(sum([i["post"]["rewrite_acc"][0] for i in metrics])/len(metrics) * 100, 2))
        if args.eval_aspect == "portability_r":
            print("Portability Reasoning: ", round(sum([i["post"]["portability"]["Reasoning_acc"][0] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "portability_s":
            print("Portability Subject_Aliasing: ", round(sum([i["post"]["portability"]["Subject_Aliasing_acc"][0] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "portability_l":
            print("Portability Logical_Generalization: ", round(sum([i["post"]["portability"]["Logical_Generalization_acc"][0] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "locality_rs":
            print("Locality Relation_Specificity: ", round(sum([i["post"]["locality"]["Relation_Specificity_acc"][0] for i in metrics])/len(metrics) * 100, 2))
        elif args.eval_aspect == "locality_f":
            print("Locality Forgetfulness: ", round(sum([i["post"]["locality"]["Forgetfulness_acc"][0] for i in metrics])/len(metrics) * 100, 2))
        else:
            raise KeyError

    
    if "fluency" in metrics[0]["post"].keys():
        print("Fluency: ", round(sum([i["post"]["fluency"]["ngram_entropy"] for i in metrics])/len(metrics) * 100, 2))

    print("\n\n")