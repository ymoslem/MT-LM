# python3 ctranslate.py <model_path> <sp_source_model> <sp_target_model> <source_file_path> <reference_file_path>

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sentencepiece as spm
import ctranslate2
import sacrebleu
from comet import download_model, load_from_checkpoint
import pandas as pd
import os
import sys
import transformers
transformers.logging.set_verbosity_error()

# Activate GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

# For debugging CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def tokenize(text, sp_source_model):
    sp = spm.SentencePieceProcessor(sp_source_model)
    tokens =sp.encode(text, out_type=str)
    tokens = [["<s>"] + tk + ["</s>"] for tk in tokens] # when upgrade CT2 revise this.
    return tokens


def detokenize(text, sp_target_model):
    sp = spm.SentencePieceProcessor(sp_target_model)
    translation = sp.decode(text)
    return translation


def translate(source_sents, model_path, sp_source_model, sp_target_model, beam_size, batch_size):
    source_sents_tok = tokenize(source_sents, sp_source_model)
    translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0,1])
    translations_tok = translator.translate_batch(source=source_sents_tok,
                                                  beam_size=beam_size,
                                                  batch_type="tokens",
                                                  max_batch_size=batch_size,
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



model_path = sys.argv[1]
sp_source_model = sys.argv[2]
sp_target_model = sys.argv[3]
source_file_path = sys.argv[4]
reference_file_path = sys.argv[5]
target_file_path = source_file_path + ".translated." + model_path.split("/")[-2]

beam_size = 5
batch_size = 4096 # 2048  # try 4096 AR-EN

# Open the source file
with open(source_file_path) as test_file:
    source_sents = test_file.readlines()
    source_sents = [sent.strip() for sent in source_sents]

# Open the reference file
with open(reference_file_path) as reference_file:
    references = reference_file.readlines()
    references = [ref.strip() for ref in references]
    print("Ref:", references[0])

# Translate the source sentences
translations = translate(source_sents, model_path, sp_source_model, sp_target_model, beam_size, batch_size)
print("MT:", translations[0])


# Save Translations
with open(target_file_path, "w+") as target_file:
    for translation in translations:
        target_file.write(translation.strip() + "\n")
    print("Translation saved at:", target_file_path)


# Calculate BLEU
bleu = sacrebleu.corpus_bleu(translations, [references], tokenize='spm')
print("BLEU:", round(bleu.score, 2))

# Calculate CHRF
chrf = sacrebleu.corpus_chrf(translations, [references])
print("CHRF:", round(chrf.score, 2))

# Calculate TER
metric = sacrebleu.metrics.TER()
ter = metric.corpus_score(translations, [references])
print("TER:", round(ter.score, 2))

# Calculate COMET
df = pd.DataFrame({"src":source_sents, "mt":translations, "ref":references})
data = df.to_dict('records')

model_path = download_model("wmt20-comet-da")
model = load_from_checkpoint(model_path)

seg_scores, sys_score = model.predict(data, batch_size=128, gpus=1)

print("COMET:", round(sys_score*100, 2))
