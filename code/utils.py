import collections
import random

import numpy as np
import torch
from transformers import AutoConfig, GPT2LMHeadModel, GPT2Model, GPT2Tokenizer


def load_model(args):
  """Creates a model and loads in weights for it."""
  config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=None)

  model = GPT2LMHeadModel.from_pretrained(
      args.model_name_or_path,
      from_tf=bool(".ckpt" in args.model_name_or_path),
    #   from_pt=True,
      config=config,
      cache_dir=None
  )
  
  model.to(args.device)
  return model

def set_seed(args, seed):
  """Set the random seed."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
