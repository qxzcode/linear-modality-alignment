import json
import readline
import copy
import re
import os
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import decode_image, ImageReadMode
from tqdm import tqdm

from adapter_module import Adapter

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('--num_adapter_tokens', type=int, default=2, help="Default: %(default)s")
parser.add_argument('--history_file', default='demo_inputs.txt', help="Default: %(default)s")
args = parser.parse_args()

try:
    readline.read_history_file(args.history_file)
except FileNotFoundError:
    pass

device = torch.device('cuda:0')



# Load the Adapter parameters:

adapter = Adapter(image_embed_dim=1024, text_embed_dim=4096, num_output_tokens=args.num_adapter_tokens)
adapter.to(device)
print(f'restoring weights from {args.checkpoint}')
adapter.load_state_dict(torch.load(args.checkpoint, map_location=device))
adapter.eval()



# Load the models and tokenizer:

print('loading CLIP model...')
device = torch.device('cuda:0')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device).vision_model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print('done')

@torch.no_grad()
def get_image_embedding(image_path: str) -> np.ndarray:
    with open(image_path, 'rb') as f:
        image_file_bytes = f.read()
    image_file_bytes = np.frombuffer(image_file_bytes, dtype=np.uint8).copy()
    image = decode_image(torch.from_numpy(image_file_bytes), ImageReadMode.RGB)
    image = clip_processor(images=image, return_tensors="pt")['pixel_values']
    outputs = clip_model(pixel_values=image.to(device), return_dict=True)
    return outputs.pooler_output.squeeze(0).numpy(force=True)

print('loading Mistral model...')
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



# # Creating the model inputs and targets:

# with torch.no_grad():
#     prefix_ids = torch.tensor([tokenizer.encode('[INST] <image:')], device=device)
#     prefix_embs = model.model.embed_tokens(prefix_ids)
#     print(f'{prefix_embs.shape=}')

# def make_input_and_targets(image_tokens: torch.FloatTensor, post_image_prompt: str = "") -> tuple[torch.FloatTensor, torch.LongTensor]:
#     (_, _, _) = image_tokens.shape

#     with device:
#         with torch.no_grad():
#             # tokenizer.pad_token_id = 0
#             # tokenizer.padding_side = 'right'
#             encoded_suffix = tokenizer.batch_encode_plus(
#                 [f'>\n\n{post_image_prompt} [/INST]'],
#                 add_special_tokens=False,
#                 return_tensors='pt',
#                 # padding=True,
#                 # truncation=True,
#                 # max_length=wandb.config.max_seq_length - 6,
#             )
#             suffix_ids = encoded_suffix['input_ids']
#             # print(suffix_ids)
#             suffix_embs = model.model.embed_tokens(suffix_ids)

#         input_embs = torch.concat([
#             prefix_embs,
#             image_tokens,
#             suffix_embs,
#         ], dim=1)

#         return input_embs



AI_OUTPUT_COLOR = '96'  # bright cyan
DIM_COLOR = '30'  # black

class ChatContext:
    """Helper class to track the chat history and run the model."""
    
    def __init__(self) -> None:
        self.first_instruction = True
        self.past_key_values = None
        self.cache_len = 0
        with device:
            self.input_embs = torch.zeros(1, 0, 4096, dtype=llm_dtype)

    def add_instruction(self, text: str):
        """Adds the given instruction text to the chat history."""

        if self.first_instruction:
            self.add_tokens([tokenizer.bos_token_id])
            self.first_instruction = False

        text = f"[INST] {text} [/INST]"

        # Replace <image:...> with adapted image embedding tokens.
        IMAGE_PATTERN = r"<image:([^>]+?)>"
        while m := re.search(IMAGE_PATTERN, text):
            self.add_text(text[:m.start()])

            print(f'\033[{DIM_COLOR}m[replacing "{m.group(1)}" with its adapted image embedding]\033[0m')
            try:
                image_index = int(m.group(1))
            except ValueError:
                image_emb = get_image_embedding('images/'+m.group(1))
            else:
                path = get_item_path(image_index)
                image_emb = np.load(path)
            image_emb = torch.from_numpy(image_emb).to(device)
            image_tokens = adapter(image_emb.unsqueeze(0))
            self.add_text("\n\n<image:")
            self.add_embeddings(image_tokens)
            self.add_text(">\n\n")

            text = text[m.end():]

        self.add_text(text)

    def add_text(self, text: str):
        """Adds the given raw text to the chat history."""
        text = text.removeprefix(' ').removesuffix(' ')
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.add_tokens(token_ids)

    @torch.no_grad()
    def add_tokens(self, token_ids: list[int]):
        """Adds the given tokens to the chat history."""
        with device:
            embeddings = model.model.embed_tokens(torch.tensor([token_ids]))
            self.add_embeddings(embeddings)

    def add_embeddings(self, embeddings: torch.FloatTensor):
        """Adds the given token embeddings to the chat history."""
        if embeddings.dtype != llm_dtype:
            embeddings = embeddings.to(llm_dtype)
        self.input_embs = torch.concat([self.input_embs, embeddings], dim=1)

    def do_ai_response(self):
        """Generate and print the model's next message."""

        print(f'\033[1;{AI_OUTPUT_COLOR}mMistral:\033[0m  ', end='', flush=True)
        generated_token_ids = []
        prev_generated_string = ""
        try:
            while True:
                # Evaluate the LLM model.
                model_outputs = model(
                    inputs_embeds=self.input_embs[:, self.cache_len:, :],
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = model_outputs.logits
                self.cache_len = self.input_embs.shape[1]
                self.past_key_values = model_outputs.past_key_values

                # Get the next predicted token ID.
                pred_token_ids = logits.argmax(dim=-1).squeeze(0).tolist()
                next_pred_token_id = pred_token_ids[-1]
                generated_token_ids.append(next_pred_token_id)
                self.add_tokens([next_pred_token_id])

                # Stop if the model emitted the end-of-sequence token.
                if next_pred_token_id == tokenizer.eos_token_id: # </s>, the EOS token
                    print()
                    break

                # Print the next part of the generated string.
                generated_string = tokenizer.decode(generated_token_ids)
                print(f'\033[{AI_OUTPUT_COLOR}m{generated_string.removeprefix(prev_generated_string)}\033[0m', end='', flush=True)
                prev_generated_string = generated_string
        except KeyboardInterrupt:
            print()
            print(f'\033[{DIM_COLOR}m[got Ctrl-C; truncated]\033[0m')
            if len(generated_token_ids) == 0 or generated_token_ids[-1] != tokenizer.eos_token_id:
                self.add_tokens([tokenizer.eos_token_id])


print()
print()
with torch.no_grad():
    with device:
        while True:
            print()
            print()
            print()
            print()
            context = ChatContext()

            while True:
                context_backup = copy.deepcopy(context)
                while True:
                    do_reset = False
                    try:
                        # Get user input.
                        instruction = input("User:     ").strip()

                        # Handle the user input.
                        if instruction == '/reset':
                            # Start a new chat.
                            do_reset = True
                            break
                        elif instruction.startswith('/tokens '):
                            # Print most similar vocab tokens.
                            image_emb = get_image_embedding('images/'+instruction.removeprefix('/tokens ').strip())
                            cosine_sim = True
                            with torch.no_grad():
                                llm_embs = model.model.embed_tokens.weight.detach().clone()
                                if cosine_sim:
                                    llm_embs /= llm_embs.norm(dim=1, keepdim=True) + 1e-7
                                adapter_embs = adapter(torch.tensor(image_emb, device=device).unsqueeze(0)).squeeze(0)
                                # adapter_embs = torch.randn_like(adapter_embs)
                                if cosine_sim:
                                    adapter_embs /= adapter_embs.norm(dim=1, keepdim=True) + 1e-7
                            dot_prods = adapter_embs @ llm_embs.T.to(adapter_embs.dtype)
                            sorts = dot_prods.argsort(dim=1, descending=True)
                            for adapter_token_index in range(adapter_embs.shape[0]):
                                index_string = f'image token {adapter_token_index}' if adapter_embs.shape[0] > 1 else 'the image token'
                                print(f'\033[{DIM_COLOR}mMost similar \033[36mLLM vocabulary tokens\033[{DIM_COLOR}m to {index_string}:\033[0m')
                                for llm_token_index in sorts[adapter_token_index][:30].tolist():
                                    dot_prod = dot_prods[adapter_token_index, llm_token_index]
                                    token_str = tokenizer.convert_ids_to_tokens(llm_token_index).replace("▁", " ")
                                    print(f'    {dot_prod:.4f}  \033[{DIM_COLOR}m{llm_token_index:>5}\033[0m  \033[1m{token_str!r}\033[0m')
                            print()
                        elif instruction == '/randvqa':
                            # Ask a random VQAv2 question.
                            with open('VQAv2/v2_OpenEnded_mscoco_val2014_questions.json') as f:
                                vqa_questions_dataset = json.load(f)
                            question = random.choice(vqa_questions_dataset['questions'])
                            image_path = f'VQAv2/val2014/COCO_val2014_{question["image_id"]:012}.jpg'
                            question_text = question['question']
                            print(f'\033[{DIM_COLOR}mquestion_id:  {question["question_id"]}\033[0m')
                            print(f'\033[{DIM_COLOR}mImage:\033[0m        {image_path}')
                            print(f'\033[{DIM_COLOR}mQuestion:\033[0m     {question_text}')
                            context.add_instruction(f'<image:../{image_path}> Given this image: {question_text}')
                            print()
                            break
                        else:
                            # Send the prompt to the model.
                            context.add_instruction(instruction)
                            print()
                            break
                    except KeyboardInterrupt:
                        print()
                        try:
                            input("Press Ctrl-C again to quit:")
                        except KeyboardInterrupt:
                            print()
                            print("Quitting...")
                            exit()
                    except Exception as e:
                        print(e)
                    context = context_backup

                if do_reset:
                    break

                # Generate and display the AI's next response.
                context.do_ai_response()
                print()

            print()
