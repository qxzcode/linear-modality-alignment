print('importing libraries...')
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
from tqdm import tqdm
from pathlib import Path
import threading
import sys
import itertools

import torch
from torchvision.io import decode_image, ImageReadMode
import numpy as np
from transformers import CLIPProcessor, CLIPModel

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
print('done')


if len(sys.argv) < 2:
    start_index = 0
else:
    [_, arg1] = sys.argv
    start_index = int(arg1)
print(f'starting at index {start_index}')
print()


print('loading CLIP model...')
device = torch.device('cuda:0')
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device).vision_model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print('done')


OUTPUT_DIR = Path('cc12m_image_embeddings/')
OUTPUT_DIR.mkdir(exist_ok=True)

def get_item_path(dataset_index: int) -> Path:
    return OUTPUT_DIR / f'{dataset_index:08}.npy'


# Setup the session to fetch images from the internet.
USER_AGENT = get_datasets_user_agent()
session = requests.Session()
session.headers.update({"user-agent": USER_AGENT})

def fetch_single_image(image_url, timeout=None, retries=0):
    """
    Fetches a single image from the internet, and decodes it to pixels.
    Returns either a torch Tensor or None, if the download failed.
    """
    for _ in range(retries + 1):
        try:
            resp = session.get(image_url, timeout=timeout)
            image = decode_image(torch.frombuffer(resp.content, dtype=torch.uint8), ImageReadMode.RGB)
            break
        except Exception as e:
            if 'Read timed out.' in str(e):
                print(e)
            image = None
    return image


# Load the dataset that contains the image URLs.
# This will auto-download it on first run.
print('loading dataset...')
dset = load_dataset("conceptual_12m")
print('done')


# Helper class to store async results.
class Future:
    def __init__(self) -> None:
        self._event = threading.Event()

    def set_value(self, value) -> None:
        self._value = value
        self._event.set()

    def get_value(self):
        self._event.wait()
        return self._value


# A queue of images to compute embeddings for as a batch. (Unused.)
embedding_queue = list[tuple[torch.Tensor, Future]]()
embed_lock = threading.RLock()
CLIP_BATCH_SIZE = 4



def process_queue():
    # Process the queue of images as a batch.
    with embed_lock:
        global embedding_queue
        embedding_queue_local = embedding_queue
        embedding_queue = []

    if len(embedding_queue_local) == 0:
        return

    images = [image for image, _ in embedding_queue_local]
    input_images = torch.concat(images, dim=0)
    import random
    token = random.random()
    print('starting', token)
    outputs = model(pixel_values=input_images.to(device), return_dict=True)
    arrays = outputs.pooler_output.numpy(force=True)
    print('ending', token)
    assert len(arrays) == len(embedding_queue_local)
    for array, (_, future) in zip(arrays, embedding_queue_local):
        future.set_value(array)

@torch.no_grad()
def get_image_embedding(image: torch.Tensor) -> np.ndarray:
    """Computes the CLIP embedding for the given image."""
    image = processor(images=image, return_tensors="pt")['pixel_values']
    outputs = model(pixel_values=image.to(device), return_dict=True)
    return outputs.pooler_output.squeeze(0).numpy(force=True)

    # This code is currently unused; I was experimenting with batching the
    # image embedding with CLIP. Unfortunately I encountered errors and no real
    # speedup so I ended up going with an unbatched approach (the code above).
    my_future = Future()
    with embed_lock:
        embedding_queue.append((image, my_future))

        if len(embedding_queue) >= CLIP_BATCH_SIZE:
            try:
                process_queue()
            except Exception as e:
                print('\n'*5)
                import traceback
                print(traceback.format_exc())
                print('\n'*5)
                raise

    return my_future.get_value()



# Set up a progress bar to track the download/processing rate.
progress = tqdm(total=len(dset['train'])-start_index, unit='image', smoothing=0.01)
progress_lock = threading.Lock()
num_good_images = 0

def process_image(index):
    """Download and embed the single image with the given index in the CC12M dataset."""

    global num_good_images

    item = dset['train'][index]
    image_url = item['image_url']
    caption = item['caption']

    # (Try to) fetch and decode the image.
    image = fetch_single_image(image_url, timeout=(3.05, 5))

    if image is not None:
        # Run the CLIP image encoder on the image.
        array = get_image_embedding(image)

        # Save the embedding vector to its file.
        output_path = get_item_path(index)
        np.save(output_path, array, allow_pickle=False)

    # Update the progress bar.
    with progress_lock:
        if image is not None:
            num_good_images += 1
            progress.set_description(f'good_images={num_good_images}/{progress.n+1}', refresh=False)
        progress.update(1)

# process_image((0, dset['train'][0]))
# quit()

NUM_THREADS = 100
assert NUM_THREADS >= CLIP_BATCH_SIZE
try:
    print('processing images...')
    # Launch a thread pool to download multiple images from different websites in parallel.
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for _ in executor.map(process_image, range(start_index, len(dset['train']))):
            pass
except KeyboardInterrupt:
    print('\n\n\nSHUTTING DOWN\n\n')
    process_queue()
    executor.shutdown(cancel_futures=True)
    exit()

progress.close()
print('all done!')
