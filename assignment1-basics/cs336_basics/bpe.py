from multiprocessing import process
import os
from typing import BinaryIO
from pretokenization_example import find_chunk_boundaries
import multiprocessing
import sys
import regex as re


def load_and_chunk_file(
    input_path: str,
    desired_num_chunks: int,
    split_special_token: list[str],): # Load the file given by "input_path" and divide it into chunks

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

        # print(type(total_frequencies))
        return total_frequencies
    
        


def pretokenize_chunk(args): # Deal with a chunk

    start,end,input_path,split_special_token = args
    
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start).decode("utf-8", errors="ignore")
        
    split_pattern = "|".join(re.escape(token) for token in split_special_token)
    text_segments = re.split(f"({split_pattern})",chunk_bytes)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    frequency_dict = {}
    for segment in text_segments:
        if segment not in split_special_token:
            # print(segment)
            for match in re.finditer(PAT,segment):
                pretoken = match.group()
                pretoken_bytes = tuple(pretoken.encode("utf-8"))
                frequency_dict[pretoken_bytes] = frequency_dict.get(pretoken_bytes,0) + 1

    # print(frequency_dict)
    return frequency_dict
    

def merge_frequencies(frequency_dict): # Calculate the frequencies from each chunks and sum them together

    total_frequencies= {}
    for every_frequency_dict in frequency_dict: # {[1,2,3,4,5]:1,[2,3,4,5,6]:2}
        for pretoken_bytes, count in every_frequency_dict.items():
            total_frequencies[pretoken_bytes] = total_frequencies.get(pretoken_bytes, 0) + count

    return total_frequencies

def initialize_vocab_and_merges(special_tokens):
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = special_tokens.encode("utf-8")

    return vocab,[]

def get_initial_pair_frequencies(frequency_dict):
    pair_freq = {}

    for pretoken_bytes, count in frequency_dict.items():
        bytes_list = [b for byte_tuple in pretoken_bytes for b in byte_tuple]

        for i in range(len(bytes_list)-1):
            pair = (bytes(bytes_list[i]), bytes(bytes_list[i+1]))
            pair_freq[pair] = pair_freq.get(pair,0) + count

    return pair_freq

def find_best_pair(pair_frequencies):
    if not pair_frequencies:
        return None

    max_freq = max(pair_frequencies.values())
    candidates = [pair for pair,freq in pair_frequencies.items() if freq == max_freq]

    return max(candidates)

def update_pair_counts(pair_frequencies, pretoken, delta_count):

    for i in range(len(pretoken) - 1):
        pair = (pretoken[i], pretoken[i + 1])
        pair_frequencies[pair] = pair_frequencies.get(pair, 0) + delta_count

        if pair_frequencies[pair] == 0:
            del pair_frequencies[pair]

def merge_pair(frequency_dict, pair_frequencies, best_pair):
    new_frequency_dict = {}
    new_pair_frequencies = pair_frequencies.copy()

    byte1, byte2 = best_pair
    merged_byte = byte1 + byte2

    for pretoken, count in frequency_dict.items():
        new_pretoken = []
        i = 0
        while (i < len(pretoken)):
            if ( i < len(pretoken) -1 and pretoken[i] == byte1 and pretoken[i+1] == byte2):
                new_pretoken.append(merged_byte)
                i += 2

            else:
                new_pretoken.append(pretoken[i])
                i += 1

        new_pretoken_tuple = tuple(new_pretoken)
        new_frequency_dict[new_pretoken_tuple] = new_frequency_dict.get(new_pretoken_tuple, 0) + count

        update_pair_counts(new_pair_frequencies, new_pretoken_tuple, count)
        update_pair_counts(new_pair_frequencies, pretoken, -count)

    return new_frequency_dict, new_pair_frequencies

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):

    frequency_dict = load_and_chunk_file(input_path,desired_num_chunks=4,split_special_token=special_tokens)
    
    vocab, merges = initialize_vocab_and_merges(special_tokens)

    pair_frequencies = get_initial_pair_frequencies(frequency_dict)

    while len(vocab) < vocab_size:
        best_pair = find_best_pair(pair_frequencies)

        if not best_pair:
            break

        frequency_dict, pair_frequencies = merge_pair(frequency_dict, pair_frequencies, best_pair)

        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token


if __name__ == "__main__":

    if sys.argv[1]=="Test":
        input_path = sys.argv[2]

        load_and_chunk_file(input_path=input_path,desired_num_chunks=4,split_special_token=["<|endoftext|>"])