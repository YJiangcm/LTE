from sentence_transformers import SentenceTransformer
import pickle
from torch.utils.data import Dataset
import os
from .ike_hparams import IKEHyperParams, IKEMultimodalHyperParams
from typing import Dict
import numpy as np


def encode_ike_facts(sentence_model: SentenceTransformer, request: Dict, hparams: IKEHyperParams):

    new_sentences = [request['prompt']]
    new_embeddings = sentence_model.encode(new_sentences) # (1, dim)
    new_facts = [request['prompt'] + " " + request['target_new']]

    base_path = f'{hparams.results_dir}/{hparams.alg_name}/embedding'
    os.makedirs(base_path, exist_ok=True)
    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]

    try:
        with open(f'{base_path}/{safe_model_name}.pkl', "rb") as fIn:
            data = pickle.load(fIn)
            existing_facts = data['sentences']
            existing_embeddings = data['embeddings']
            updated_facts = existing_facts + new_facts
            updated_embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
    except FileNotFoundError:
        updated_facts = new_facts
        updated_embeddings = new_embeddings
    
    # assert len(updated_facts) == updated_embeddings.shape[0]

    with open(f'{base_path}/{safe_model_name}.pkl', "wb") as fOut:
        pickle.dump({'sentences': updated_facts, 'embeddings': updated_embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
        
def encode_ike_facts_multimodal(sentence_model: SentenceTransformer, ds: Dataset, hparams: IKEMultimodalHyperParams):

    sentences = []
    for i, train_data in enumerate(ds):
        new_fact = train_data['prompt'] + ' ' + train_data['target']
        target_new = train_data['target']
        paraphrases = train_data['rephrase_prompt']
        neighbors = train_data['locality_prompt']
        neighbors_ans = train_data['locality_ground_truth']
        sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n")
        sentences.append(f"New Fact: {new_fact}\nPrompt: {paraphrases} {target_new}\n\n")
        sentences.append(f"New Fact: {new_fact}\nPrompt: {neighbors} {neighbors_ans}\n\n")


    embeddings = sentence_model.encode(sentences)
    base_path = f'{hparams.results_dir}/{hparams.alg_name}/embedding'
    os.makedirs(base_path, exist_ok=True)
    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{base_path}/{hparams.task_name}_embeddings.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)
