from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import numpy as np

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(150)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate_sequence(model, length, context, token_type_ids, attention_mask, num_samples=1, temperature=1.0, top_k=0,
                      top_p=0.0, repetition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    token_type_ids = token_type_ids.repeat(num_samples, 1)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    attention_mask = attention_mask.repeat(num_samples, 1)
    generated = context
    generated = model.generate(input_ids=generated,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               max_length=length + generated.size(1),
                               temperature=temperature,
                               top_k=top_k,
                               do_sample=True,
                               num_return_sequences=num_samples,
                               repetition_penalty=repetition_penalty,
                               top_p=top_p,
                               )
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='model/all_15/final_tr0.3454456596468458/', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument('--tokenizer_path', default='model/pretrained/', type=str, required=False, help='选择词库')
    parser.add_argument('--special_tokens', default='data/special_tokens.txt', type=str, required=False,
                        help='special_tokens')
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=110)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>',
                        help="Token at which text generation is stopped")
    parser.add_argument('--input_file', type=str, default='data/test.soloist.json', help="input json file to decoding")
    parser.add_argument('--output_dir', type=str, default='output/', help="save path")
    parser.add_argument('--max_turn', type=int, default=15, help="number of turns used as context")
    parser.add_argument('--batch_size', type=int, default=10, help="number of batch size")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    if args.special_tokens:
        for line in open(args.special_tokens, 'r', encoding='utf-8'):
            line = line.strip('\n')
            tokenizer.add_tokens(line)
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)

    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])
    induction_id = tokenizer.convert_tokens_to_ids(['=>'])

    logger.info(f'Start decode {args.input_file}...')
    inputs = json.load(open(args.input_file))
    output_tests = []

    for idx in tqdm(range(0, len(inputs), args.batch_size), desc=f'inputs'):
        examps = inputs[idx: idx + args.batch_size]
        context_tokens = []
        token_type_ids = []
        attention_masks = []
        length = 0
        for examp in examps:
            history = examp['history']
            context = history[-args.max_turn:]
            context_ids = []
            token_ids_for_context = []
            attention_mask = []
            for cxt in context:
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                context_ids += ids
                attention_mask += [1] * len(ids)
                if 'user :' in cxt:
                    token_ids_for_context += user_token_id * len(ids)
                else:
                    token_ids_for_context += system_token_id * len(ids)

            context_token = context_ids + induction_id
            token_type_id = token_ids_for_context + system_token_id
            attention_mask = attention_mask + [1]
            length = len(context_token) if len(context_token) > length else length
            context_tokens.append(context_token)
            token_type_ids.append(token_type_id)
            attention_masks.append(attention_mask)

        for token_idx in range(len(context_tokens)):
            context_token = context_tokens[token_idx]
            if len(context_token) < length:
                token_type_ids[token_idx] = [0] * (length - len(context_token)) + token_type_ids[token_idx]
                attention_masks[token_idx] = [0] * (length - len(context_token)) + attention_masks[token_idx]
                context_tokens[token_idx] = [tokenizer.pad_token_id] * (length - len(context_token)) + context_tokens[token_idx]
        out = generate_sequence(
            model=model,
            context=context_tokens,
            token_type_ids=token_type_ids,
            attention_mask=attention_masks,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )
        text_list = tokenizer.batch_decode(out, skip_special_tokens=True)
        count = 0
        examples = []
        for text in text_list:
            examples.append(text)
            count += 1
            if count == args.num_samples:
                output_tests.append(examples)
                examples = []
                count = 0
    model_str = args.model_name_or_path.split('/')[1]
    json.dump(output_tests, open(args.output_dir + f'{model_str}.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
