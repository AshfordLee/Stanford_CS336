from cs336_basics.bpe import train_bpe
from loguru import logger
from tests.common import gpt2_bytes_to_unicode
import json
from pathlib import Path

# uv run python train_bpe_tinystories.py 

# logger.add('/root/autodl-tmp/Stanford_CS336/Logs/Assignment1.log')
logger.add('./../Logs/Assignment1.log')


def save_vocab_merge(vocab, merges, output_path='./../../TinyStories_Result'):
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)

    byte_to_unicode = gpt2_bytes_to_unicode()

    # Vocab:{id:bytes} -> {unicode_string:id}
    vocab_unicode = {}

    for token_id,token_bytes in vocab.items():
        unicode_chars = [byte_to_unicode[b] for b in token_bytes]
        unicode_string = ''.join(unicode_chars)
        vocab_unicode[unicode_string] = token_id

    vocab_path = output_dir/"vocab.json"

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_unicode, f, ensure_ascii=False, indent=2)

    merge_path = output_dir/"merges.txt"
    
    # Merge:[tuple(bytes,bytes)]
    with open(merge_path,'w',encoding='utf-8') as f:
        for merge_pair in merges:
            token1_bytes, token2_bytes = merge_pair
            token1_unicode = ''.join(byte_to_unicode[b] for b in token1_bytes)
            token2_unicode = ''.join(byte_to_unicode[b] for b in token2_bytes)
            f.write(f"{token1_unicode} {token2_unicode}\n")

    return vocab_path,merge_path

def train_bpe_tinystories():
    
    input_path = "./../../data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000

    vocab, merges = train_bpe(input_path=input_path,vocab_size=vocab_size,special_tokens=special_tokens,Debug=False)

    logger.info(f"Vocab:{vocab}")

    logger.info(f"Merges:{merges}")


    vocab_path, merges_path = save_vocab_merge(vocab, merges)
    logger.info(f"Vocabulary saved to: {vocab_path}")
    logger.info(f"Merges saved to: {merges_path}")


if __name__ == "__main__":
    train_bpe_tinystories()