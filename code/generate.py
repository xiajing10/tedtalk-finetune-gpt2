import collections
import random
import json
import sys

import numpy as np
import torch
from dict_obj import obj
import yaml
from utils import set_seed
from transformers import AutoConfig, GPT2LMHeadModel, GPT2Model, GPT2Tokenizer



def generate(args, tokenizer, model, prompt_text):
    """Generating sampling for the provided prompt using the provided model."""
    set_seed(args, args.seed)

    #   tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=None)

    #   requires_preprocessing = args.model_type in run_generation.PREPROCESSING_FUNCTIONS.keys()
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Remove the excess text that was used for pre-processing
        text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]

        # Add the prompt at the beginning of the sequence.
        if args.include_prompt:
            total_sequence = prompt_text + text
        else:
            total_sequence = text

        generated_sequences.append(total_sequence)

    return generated_sequences


def main():
    if len(sys.argv) < 2:
        print('Missing config file.')
    else:
        config_path = sys.argv[1]

    data = yaml.load(open(config_path, 'r'),Loader=yaml.Loader)
    CHECKPOINT_PATH = data['model_name_or_path']
    output_dir = data['output_dir']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("Running on device: ", device)

    args = collections.defaultdict(
        model_name_or_path=CHECKPOINT_PATH,
        output_dir=output_dir,
        n_gpu=n_gpu,
        mlm=False,
        device=device,
        model_type='gpt2',
        seed=42,
        stop_token=None, # Set this if your dataset has a special word that indicates the end of a text.
        temperature=1.0,  # temperature sampling. Set this to temperature=1.0 to not use temperature.
        k=50,  # k for top-k sampling. Set this to k=0 to not use top-k.
        p=1.0,  # p for nucleus sampling. Set this to p=1.0 to not use nucleus sampling.
        repetition_penalty=None,
        length=50,  # Number of tokens to generate.
        num_return_sequences=3,
        include_prompt = data['include_prompt']
    )

    args = obj(args)

    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_PATH)
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_PATH, pad_token_id=tokenizer.eos_token_id)
    model.to(args.device)
    prompt = open(data['prompt'], 'r').read()
    if data['user_input']:
        prompt += data['user_input']

    sequences = generate(args, tokenizer, model, prompt)
    samples = {idx: sequence for idx, sequence in enumerate(sequences)}
    with open(output_dir+'samples.txt', 'w') as f:
        json.dump(samples, f)
    print("Generated result is saved at : ", output_dir+'samples.txt')

if __name__ == "__main__":
    main()