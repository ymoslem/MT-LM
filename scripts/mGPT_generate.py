# python3 gpt_generate_beam.py <target_file> <new_output_file> <lang> <gpu_id>

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from tqdm import tqdm
import pysbd
import os
import sys

# Which GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[4]

# For debugging CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Input and output files
target_file = sys.argv[1]
output_file = sys.argv[2]
language = sys.argv[3]

# Define sentence segmenter
segmenter = pysbd.Segmenter(language=language, clean=True)

# Load  the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/mGPT")
model =  AutoModelForCausalLM.from_pretrained("sberbank-ai/mGPT",
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True,
                                              pad_token_id=tokenizer.eos_token_id)
model = model.half()
model = model.to("cuda")
print("Model loaded.")

# [Optional] Set seed to reproduce results.
# set_seed(0)


def line_count(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

tqdm_total = line_count(target_file)
print("Line count:", tqdm_total)

# Generate sentences
with open(target_file) as target, open (output_file, "a+") as output:
    output.seek(0)
    output.truncate()

    for line in tqdm(target, total=tqdm_total):
        line = line.strip()
        input_ids = tokenizer(line, return_tensors="pt").input_ids.to("cuda")

        generated_tokens = model.generate(input_ids,
                                          do_sample=True,
                                          max_length=300,
                                          top_k=50,
                                          top_p=0.95,
                                          num_return_sequences=5,
                                          early_stopping=True)

        generated_text_beam = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for genenrated_output in generated_text_beam:
            # Split generated text into sentences
            generated_lines = segmenter.segment(genenrated_output)
            # remove the first sent, which is the original one
            # remove the last sent as it might be truncated
            generated_lines = generated_lines[1:-1]
            # Add new lines between sentences, and exclude too short strings (less than 10 characters)
            generated_lines = "\n".join([text.strip() for text in generated_lines if len(text) > 10]) + "\n"

            output.write(generated_lines)

print("Done!")
