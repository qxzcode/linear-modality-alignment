import os
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from adapter_module import Adapter

from argparse import ArgumentParser

hyperparam_defaults = {
    "learning_rate": 0.003,
    "batch_size": 4,
    "num_iters": 100_000,
    "grad_clip_norm": 1.0,
    "num_adapter_tokens": 2,
    "max_seq_length": 100,
    "grad_accum_steps": 1,
}

parser = ArgumentParser()
parser.add_argument('-r', '--resume_checkpoint', default=None)
parser.add_argument("--no-wandb", action="store_true")
for name, default_value in hyperparam_defaults.items():
    parser.add_argument(
        f"--{name}",
        type=type(default_value),
        default=default_value,
        help="Default: %(default)s",
    )
args = parser.parse_args()

device = torch.device('cuda:0')

# Configure Weights & Biases.
wandb.init(
    entity='qxz',
    project="nlp-term-project",
    config={
        "device": str(device),
        **{name: getattr(args, name) for name in hyperparam_defaults.keys()},
    },
    mode="disabled" if args.no_wandb else "online",
)



# Load the model and tokenizer:

print('loading model...')
llm_dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(  # MistralForCausalLM
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    torch_dtype=llm_dtype,
)

print('loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'
print('done')



# Load the dataset:
print('loading dataset...')
start = time.perf_counter()
dset = load_dataset("conceptual_12m")['train']
print('took', time.perf_counter()-start, 'sec')

IMG_EMBEDDINGS_DIR = Path('cc12m_image_embeddings/')
def get_item_path(dataset_index: int) -> Path:
    return IMG_EMBEDDINGS_DIR / f'{dataset_index:08}.npy'

print('getting list of computed embeddings...')
start = time.perf_counter()
available_indices = [
    int(filename.removesuffix('.npy'))
    for filename in os.listdir(IMG_EMBEDDINGS_DIR)
]
print(f'{len(available_indices)=}')
print('took', time.perf_counter()-start, 'sec')

def sample_batch(batch_size: int) -> list[tuple[np.ndarray, str]]:
    batch = []
    while len(batch) < batch_size:
        index = random.choice(available_indices)
        path = get_item_path(index)
        try:
            emb = np.load(path)
        except FileNotFoundError:
            pass
        except Exception:
            import traceback
            traceback.print_exc()
        else:
            batch.append((emb, dset[index]['caption'].strip()))
    return batch



# Creating the model inputs and targets:

with torch.no_grad():
    prompt_prefix = ""
    prompt_suffix = "\n"

    prefix_ids = torch.tensor([tokenizer.encode(prompt_prefix)], device=device)
    prefix_embs = model.model.embed_tokens(prefix_ids)
    print(f'{prefix_embs.shape=}')

    prompt_suffix_ids = torch.tensor([tokenizer.encode(prompt_suffix, add_special_tokens=False)], device=device)

    PERSON_TOKENS: set[int] = set(tokenizer.encode("<PERSON>", add_special_tokens=False))

def make_input_and_targets(image_tokens: torch.FloatTensor, captions: list[str]) -> tuple[torch.FloatTensor, torch.LongTensor]:
    assert isinstance(captions, list)
    (batch_size, _, _) = image_tokens.shape
    assert len(captions) == batch_size

    with device:
        with torch.no_grad():
            tokenizer.pad_token_id = 0
            tokenizer.padding_side = 'right'
            encoded_suffix = tokenizer.batch_encode_plus(
                [f"{caption}</s>" for caption in captions],
                add_special_tokens=False,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=wandb.config.max_seq_length - (prefix_embs.shape[1] + image_tokens.shape[1] + prompt_suffix_ids.shape[1]),
            )
            suffix_ids = torch.concat([
                torch.tile(prompt_suffix_ids, [batch_size, 1]),
                encoded_suffix['input_ids'][:, :-1], # Strip off the </s> token, which shifts the targets one to the left.
            ], dim=1)
            suffix_embs = model.model.embed_tokens(suffix_ids)

        input_embs = torch.concat([
            torch.tile(prefix_embs, [batch_size, 1, 1]),
            image_tokens,
            suffix_embs,
        ], dim=1)

        with torch.no_grad():
            target_ids = encoded_suffix['input_ids']
            target_ids[encoded_suffix['attention_mask'] == 0] = -1
            for token_id in PERSON_TOKENS:
                target_ids[target_ids == token_id] = -1

        return input_embs, target_ids



# Training loop:

adapter = Adapter(image_embed_dim=1024, text_embed_dim=4096, num_output_tokens=wandb.config.num_adapter_tokens)
adapter.train()
adapter.to(device)
if args.resume_checkpoint is not None:
    print(f'restoring weights from {args.resume_checkpoint}')
    adapter.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))

optimizer = torch.optim.Adam(adapter.parameters(), lr=wandb.config.learning_rate)

print()
print()
for iter_num in tqdm(range(1, wandb.config.num_iters+1), unit='iter'):
    # Get a batch of data.
    start = time.perf_counter()
    batch = sample_batch(batch_size=wandb.config.batch_size)

    # Run the adapter model to get the image "tokens".
    image_embs = torch.from_numpy(np.stack([emb for emb, _ in batch])).to(device)
    image_tokens = adapter(image_embs)

    # Get the LLM inputs and targets.
    captions = [caption for _, caption in batch]
    input_embs, target_ids = make_input_and_targets(image_tokens, captions)
    print('batch seq len:', input_embs.shape[1])

    # Compute token logits using the LLM.
    logits = model(
        inputs_embeds=input_embs.to(llm_dtype),
        return_dict=True,
    ).logits

    # Compute the loss.
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits[:, -target_ids.shape[1]:, :].reshape(-1, vocab_size), target_ids.reshape(-1), ignore_index=-1)
    print(f'loss={loss.item()}')

    metrics = {
        'loss': loss,
        'batch_seq_len': input_embs.shape[1],
    }

    if iter_num % wandb.config.grad_accum_steps == 0:
        # Update the parameters.
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=wandb.config.grad_clip_norm, error_if_nonfinite=True)
        print(f'grad_norm={grad_norm.item()}')
        metrics['grad_norm'] = grad_norm
        optimizer.step()

    # Log metrics to Weights & Biases.
    wandb.log(metrics)

    # Save checkpoints.
    checkpoint_frequency = 100 if iter_num < 500 else 500
    if iter_num % checkpoint_frequency == 0:
        print('saving checkpoint')
        torch.save(adapter.state_dict(), 'checkpoints/adapter_latest.pt')
        torch.save(adapter.state_dict(), f'checkpoints/adapter_iter{iter_num:08}.pt')
