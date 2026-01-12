from multiprocessing import process
import os
from typing import BinaryIO
from .pretokenization_example import find_chunk_boundaries
# from pretokenization_example import find_chunk_boundaries
import multiprocessing
import sys
import regex as re
from loguru import logger
from collections import Counter,defaultdict
import time
import pprint

logger.add('/root/autodl-tmp/Stanford_CS336/Logs/Assignment1.log')

def load_and_chunk_file(
    input_path: str,
    desired_num_chunks: int,
    split_special_token: list[str],
    Debug=False): # Load the file given by "input_path" and divide it into chunks

    with open(input_path,"rb") as f:
        num_processes = 4

        split_token_bytes = split_special_token[0].encode("utf-8")
        boundaries = find_chunk_boundaries(file=f,desired_num_chunks=desired_num_chunks,split_special_token=split_token_bytes)

        # find all chunks as a list
        # processing of each chunk are implemented in the next func

    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_args=[(start,end,input_path,split_special_token) for start,end in zip(boundaries[:-1],boundaries[1:])]
        results=pool.map(pretokenize_chunk,chunk_args)


        total_frequencies = merge_frequencies(results)

        if Debug==True:
            logger.info(f"Frequency_Dict:\n{pprint.pformat(total_frequencies, indent=2)}")
        return total_frequencies
    
        


def pretokenize_chunk(args): # Deal with a chunk

    start,end,input_path,split_special_token = args
    
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start).decode("utf-8", errors="ignore")
        
    split_pattern = "|".join(re.escape(token) for token in split_special_token)
    text_segments = re.split(f"({split_pattern})",chunk_bytes)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # frequency_dict = {}
    frequency_dict = defaultdict(int)
    for segment in text_segments:
        if segment not in split_special_token:
            # print(segment)
            for match in re.finditer(PAT,segment):
                pretoken = match.group()
                pretoken_bytes = pretoken.encode("utf-8")
                pretoken_bytes_tuple = tuple(bytes([b]) for b in pretoken_bytes)
                # frequency_dict[pretoken_bytes_tuple] = frequency_dict.get(pretoken_bytes_tuple,0) + 1
                frequency_dict[pretoken_bytes_tuple] += 1

    # logger.info(f"Frequency_Dict:{frequency_dict}")
    return dict(frequency_dict)
    

def merge_frequencies(frequency_dict): # Calculate the frequencies from each chunks and sum them together

    # total_frequencies= {}
    total_frequencies = defaultdict(int)
    for every_frequency_dict in frequency_dict: # {[1,2,3,4,5]:1,[2,3,4,5,6]:2}
        for pretoken_bytes, count in every_frequency_dict.items():
            # total_frequencies[pretoken_bytes] = total_frequencies.get(pretoken_bytes, 0) + count
            total_frequencies[pretoken_bytes] += count

    return total_frequencies

def initialize_vocab_and_merges(special_tokens):
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab,[]

def get_initial_pair_frequencies(frequency_dict,Debug=False):
    pair_freq = defaultdict(int)
    pair_to_tokens = defaultdict(set)

    for pretoken_bytes, count in frequency_dict.items():
        # print(pretoken_bytes)
        # print(count)
        # bytes_list = [b for byte_tuple in pretoken_bytes for b in byte_tuple]


        for i in range(len(pretoken_bytes)-1):
            # pair = (pretoken_bytes[i:i+1],pretoken_bytes[i+1:i+2])
            pair = ((pretoken_bytes[i],), (pretoken_bytes[i+1],))
            pair_freq[pair] = pair_freq.get(pair,0) + count
            pair_to_tokens[pair].add(pretoken_bytes)

    if Debug==True:
        logger.info(f"Pair_Freq:\n{pprint.pformat(pair_freq, indent=2)}")
        logger.info(f"Pair_To_Tokens:\n{pprint.pformat(pair_to_tokens, indent=2)}")

    return pair_freq, pair_to_tokens

def find_best_pair(pair_frequencies):
    if not pair_frequencies:
        return None

    best_pair = tuple()
    max_freq = -1

    for pair,freq in pair_frequencies.items():
        if freq > max_freq:
            max_freq = freq
            best_pair = pair

        elif freq == max_freq:
            best_pair = max(best_pair,pair)

    return best_pair

def update_pair_counts(pair_frequencies, pretoken, delta_count):

    for i in range(len(pretoken) - 1):
        pair = (pretoken[i:i+1], pretoken[i+1:i+2])
        pair_frequencies[pair] = pair_frequencies.get(pair, 0) + delta_count

        if pair_frequencies[pair] == 0:
            del pair_frequencies[pair]

# def merge_pair(frequency_dict, pair_frequencies, pair_to_tokens, best_pair):

#     # logger.info(f"Frequency_Dict:{frequency_dict}")
#     # logger.info(f"Pair_Frequencies:{pair_frequencies}")
#     # logger.info(f"Best_Pair:{best_pair}")

#     # new_frequency_dict = {}
#     # new_pair_frequencies = pair_frequencies.copy() 

#     new_frequency_dict = defaultdict(int)  
#     new_pair_frequencies = defaultdict(int, pair_frequencies.copy())


#     byte1_tuple, byte2_tuple = best_pair
#     merged_byte = byte1_tuple[0] + byte2_tuple[0]
#     # logger.info(f"Merged_Byte:{merged_byte}，Merged_byte_type:{type(merged_byte)}")

#     for pretoken, count in frequency_dict.items():
#         new_pretoken_list = []
#         i = 0
#         while (i <= len(pretoken) - 1):
#             # if ( i < len(pretoken) - 1 and pretoken[i:i+1] == byte1_tuple and pretoken[i+1:i+2] == byte2_tuple):
#             if (i < len(pretoken) -1 and (pretoken[i],) == byte1_tuple and (pretoken[i+1],) == byte2_tuple):
#                 new_pretoken_list.append(merged_byte)
#                 i += 2

#             else:
#                 new_pretoken_list.append(pretoken[i])
#                 i += 1

#         new_pretoken_tuple = tuple(new_pretoken_list)
#         # logger.info(f"pretoken:{pretoken},pretoken_type:{type(pretoken)}")
#         # logger.info(f"new_protoken_tuple:{new_pretoken_tuple}")
#         # new_frequency_dict[new_pretoken_tuple] = new_frequency_dict.get(new_pretoken_tuple, 0) + count
#         new_frequency_dict[new_pretoken_tuple] += count 

#         # update_pair_counts(new_pair_frequencies, new_pretoken_tuple, count)
#         # update_pair_counts(new_pair_frequencies, pretoken, -count)

#         if new_pretoken_tuple != pretoken:
            
#             for i in range(len(pretoken)-1):
#                 # old_pair = (pretoken[i:i+1], pretoken[i+1:i+2])
#                 old_pair = ((pretoken[i],), (pretoken[i+1],)) 
#                 # new_pair_frequencies[old_pair] = new_pair_frequencies.get(old_pair, 0) - count
#                 new_pair_frequencies[old_pair] -= count
#                 if new_pair_frequencies[old_pair] <= 0:
#                     new_pair_frequencies.pop(old_pair, None)

#             for i in range(len(new_pretoken_tuple) - 1):
#                 # new_pair = (new_pretoken_tuple[i:i+1], new_pretoken_tuple[i+1:i+2])
#                 new_pair = ((new_pretoken_tuple[i],), (new_pretoken_tuple[i+1],)) 
#                 # new_pair_frequencies[new_pair] = new_pair_frequencies.get(new_pair, 0) + count
#                 new_pair_frequencies[new_pair] += count

#     # logger.info(f"New_Frequency_Dict:{new_frequency_dict}")
#     # logger.info(f"New_Pair_Frequencies:{new_pair_frequencies}")

#     return dict(new_frequency_dict), dict(new_pair_frequencies)


def merge_pair(frequency_dict, pair_frequencies, pair_to_tokens, best_pair,Debug=False):

    byte1_tuple, byte2_tuple = best_pair
    merged_byte = byte1_tuple[0] + byte2_tuple[0]

    affected_tokens = pair_to_tokens.get(best_pair,set()).copy()

    if best_pair in pair_to_tokens:
        del pair_to_tokens[best_pair]

    for pretoken in affected_tokens:
        count = frequency_dict[pretoken]

        frequency_dict[pretoken] -= count
        if frequency_dict[pretoken] <= 0:
            del frequency_dict[pretoken]

        
        new_pretoken_list = []
        i = 0
        while i < len(pretoken):
            if (i < len(pretoken) - 1 and 
                (pretoken[i],) == byte1_tuple and 
                (pretoken[i+1],) == byte2_tuple):
                new_pretoken_list.append(merged_byte)
                i += 2
            else:
                new_pretoken_list.append(pretoken[i])
                i += 1

        new_pretoken_tuple = tuple(new_pretoken_list)
        frequency_dict[new_pretoken_tuple] += count


        if new_pretoken_tuple != pretoken:

            for i in range(len(pretoken) - 1):
                old_pair = ((pretoken[i],), (pretoken[i+1],))
                pair_frequencies[old_pair] -= count
                if pair_frequencies[old_pair] <= 0:
                    del pair_frequencies[old_pair]

                if old_pair in pair_to_tokens:
                    pair_to_tokens[old_pair].discard(pretoken)
                    if not pair_to_tokens[old_pair]:
                        del pair_to_tokens[old_pair]
            

            for i in range(len(new_pretoken_tuple) - 1):
                new_pair = ((new_pretoken_tuple[i],), (new_pretoken_tuple[i+1],))
                pair_frequencies[new_pair] += count

                pair_to_tokens[new_pair].add(new_pretoken_tuple)

    if Debug==True:
        logger.info(f"Frequency_Dict:\n{pprint.pformat(frequency_dict, indent=2)}")
        logger.info(f"Pair_Freq:\n{pprint.pformat(pair_frequencies, indent=2)}")
        logger.info(f"Pair_To_Tokens:\n{pprint.pformat(pair_to_tokens, indent=2)}")

    return frequency_dict, pair_frequencies, pair_to_tokens

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str], Debug=False):

    total_start = time.time()
    logger.info("=== Start Training BPE ===")



    load_start = time.time()
    if Debug==True:
        frequency_dict = load_and_chunk_file(input_path, desired_num_chunks=4, split_special_token=special_tokens, Debug=True)
    else:
        frequency_dict = load_and_chunk_file(input_path, desired_num_chunks=4, split_special_token=special_tokens)
    load_time = time.time() - load_start
    logger.info(f"文件加载和分块耗时: {load_time:.4f}秒")
    
    init_start = time.time()
    vocab, merges = initialize_vocab_and_merges(special_tokens)
    init_time = time.time() - init_start
    logger.info(f"初始化词表耗时: {init_time:.4f}秒")

    pair_start = time.time()
    if Debug==True:
        pair_frequencies, pair_to_tokens = get_initial_pair_frequencies(frequency_dict,Debug=True)
    else:
        pair_frequencies, pair_to_tokens = get_initial_pair_frequencies(frequency_dict)
    pair_time = time.time() - pair_start
    logger.info(f"计算初始pair频率耗时: {pair_time:.4f}秒")
    

    loop_start = time.time()

    while len(vocab) < vocab_size:
        best_pair = find_best_pair(pair_frequencies)

        # logger.info(f"Best Pair:\n{pprint.pformat(best_pair, indent=2)}")
        if not best_pair:
            break

        if Debug:
            frequency_dict, pair_frequencies, pair_to_tokens = merge_pair(frequency_dict, pair_frequencies, pair_to_tokens, best_pair,Debug=True)

        else:
            frequency_dict, pair_frequencies, pair_to_tokens = merge_pair(frequency_dict, pair_frequencies, pair_to_tokens, best_pair,Debug=False) 
        merges.append((best_pair[0][0],best_pair[1][0]))

        new_token = best_pair[0][0] + best_pair[1][0]
        vocab[len(vocab)] = new_token


    loop_end = time.time() - loop_start
    logger.info(f"计算循环耗时: {loop_end:.4f}秒")
    return vocab,merges


if __name__ == "__main__":

    if sys.argv[1]=="Test":
        input_path = sys.argv[2]

        # load_and_chunk_file(input_path=input_path,desired_num_chunks=4,split_special_token=["<|endoftext|>"])
        train_bpe(input_path = input_path,vocab_size=500,special_tokens=["<|endoftext|>"],Debug=True)
