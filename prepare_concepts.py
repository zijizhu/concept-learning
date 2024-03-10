import os
import clip
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader


dataset2prompt_prefix = {'CUB_200_2011': 'The bird has '}


@torch.inference_mode()
def encode_concepts(model, texts: list[str], prompt_prefix: str,
                    output_dir: str, batch_size: int, rescale=True, device='cpu'):
    encoded = []   # Matrix T

    num_batches = len(texts) // batch_size + 1
    for i in range(num_batches):
        batch_concepts = texts[i * batch_size: (i + 1) * batch_size]
        batch_concepts_token = clip.tokenize([prompt_prefix + attr for attr in batch_concepts]).to(device)
        encoded.append(model.encode_text(batch_concepts_token))

    encoded = torch.cat(encoded).detach().cpu()
    # Rescale each row to a unit vector
    if rescale:
        encoded /= encoded.norm(p=2, dim=-1, keepdim=True)

    return encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--concept_path', type=str)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--backbone', default='RN50', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='clip', choices=['clip', 'open-clip'], type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], type=str)

    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Load encoder model
    if args.model == 'clip':
        encoder, preprocessor = clip.load(args.backbone)
    elif args.model == 'open_clip':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # Load data
    concept_texts = open(args.concept_path, 'r').read().strip().split("\n")

    # Encode Concepts
    print('Encoding concepts...')
    encoded = encode_concepts(encoder,
                              concept_texts,
                              dataset2prompt_prefix[args.dataset],
                              args.output_dir,
                              args.batch_size,
                              device=args.device)
    torch.save(encoded, os.path.join(args.output_dir, f'concepts_{args.model}_{args.backbone}.pt'))
