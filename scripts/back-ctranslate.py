# python3 ctranslate.py <model_path> <sp_source_model> <sp_target_model> <source_file_path>

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sentencepiece as spm
import ctranslate2
import sacrebleu
from tqdm import tqdm
import sys
import os

# Which GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# For debugging CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def tokenize(text, sp_source_model):
    sp = spm.SentencePieceProcessor(sp_source_model)
    tokens =sp.encode(text, out_type=str)
    # tokens = [[">>ara<<"] + tk for tk in tokens]  # for EN-AR OPUS model
    return tokens


def detokenize(text, sp_target_model):
    sp = spm.SentencePieceProcessor(sp_target_model)
    translation = sp.decode(text)
    return translation


def translate(source_sents, model_path, sp_source_model, sp_target_model, beam_size):
    source_sents_tok = tokenize(source_sents, sp_source_model)
    translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0,1])
    translations_tok = translator.translate_batch(source=source_sents_tok,
                                                  beam_size=beam_size,
                                                  batch_type="tokens",
                                                  max_batch_size=4096,
                                                  replace_unknowns=True)
    translations = [detokenize(translation[0]["tokens"], sp_target_model) for translation in translations_tok]
    return translations


def translate_with_prefix(source_sents, prefix_phrases, model_path, sp_source_model, sp_target_model, beam_size):
    source_sents_tok = tokenize(source_sents, sp_source_model)
    prefix_phrases_tok = tokenize(prefix_phrases, sp_target_model)
    translator = ctranslate2.Translator(model_path, "cuda")
    translations_tok = translator.translate_batch(source=source_sents_tok,
                                                  target_prefix=prefix_phrases_tok,
                                                  num_hypotheses=10,
                                                  return_alternatives=True,
                                                  beam_size=beam_size)
    translations = [detokenize(translation[0]["tokens"], sp_target_model) for translation in translations_tok]
    return translations


# Function to split source lines into chunks to avoid out-of-memory errors
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


model_path = sys.argv[1]
sp_source_model = sys.argv[2]
sp_target_model = sys.argv[3]
source_file_path = sys.argv[4]
target_file_path = source_file_path + ".translated"

beam_size = 5

with open(source_file_path) as source:
    source_sents = source.readlines()
    print("text loaded.")


# Translate and save Translations
with open(target_file_path, "a+") as target:
    target.seek(0)
    target.truncate()

    chunk_size = 500

    tqdm_total = round(len(source_sents)/chunk_size)
    for source_chunck in tqdm(chunks(source_sents, chunk_size), total=tqdm_total):
        translations = translate(source_chunck, model_path, sp_source_model, sp_target_model, beam_size)
        translations = "\n".join([translation.strip() for translation in translations]) + "\n"

        target.write(translations)

print("Done! File saved at", target_file_path)
