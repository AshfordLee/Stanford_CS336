from collections import defaultdict
import regex as re
import json
from typing import Iterable
from loguru import logger

logger.add('./../Logs/Assignment1.log')
class tokenizer():

    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.vocab_reverse = {token_bytes:token_id for token_id,token_bytes in self.vocab.items()}
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls,vocab_filepath:str,merges_filepath:str,special_tokens:list[str]=None):
        with open(file=vocab_filepath,mode='r',encoding='utf-8') as f:
            vocab_unicode = json.load(f)

        vocab = {}
        for unicode_str,token_id in vocab_unicode.items():
            vocab[token_id] = unicode_str.encode('utf-8')


        merges = []
        with open(file=merges_filepath,mode='r',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    token1_str, token2_str = line.strip().split()

                    merges.append((token1_str.encode('utf-8'),token2_str.encode('utf-8')))

        return cls(vocab,merges,special_tokens)


    def encode(self,text:str)->list[int]:
        # logger.info(f"Start encoding text:{text[0]}")
        # print(f"Start encoding text:{text[0]}")
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_specials)

            parts = re.split(f"({special_pattern})", text)

            result = []

            for part in parts:
                if not part:
                    continue

                elif part in self.special_tokens:
                    special_bytes = part.encode("utf-8")
                    token_id = self.vocab_reverse.get(special_bytes,None)
                    if token_id is not None:
                        result.append(token_id)

                else:
                    part_result = self.encode_normal_part(part)
                    result.extend(part_result)
            return result

        else:
            return self.encode_normal_part(text)

    def encode_normal_part(self,text:str)->list[int]:
        # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = []
        for match in re.finditer(self.PAT,text):
            pre_tokens.append(match.group())

        result = []

        for pre_token in pre_tokens:
            pre_token_bytes = pre_token.encode("utf-8")
            token_parts = [bytes([b]) for b in pre_token_bytes]

            merged_parts = self.apply_bpe_merges(token_parts)

            for part in merged_parts:
                token_id = self.vocab_reverse.get(part,None)
                if token_id is not None:
                    result.append(token_id)



        return result

    def apply_bpe_merges(self,token_parts)->list[bytes]:
        # logger.info(f"merges:")
        current_parts = token_parts.copy()

        for merge_pair in self.merges:
            byte1,byte2 = merge_pair

            i=0
            while i < len(current_parts) - 1:
                if current_parts[i]==byte1 and current_parts[i+1]==byte2:
                    merged = byte1 + byte2
                    current_parts[i] = merged
                    del current_parts[i+1]

                else:
                    i += 1

        return current_parts



        

    def encode_iterable(self,iterable:Iterable[str])->Iterable[int]:
        for text in iterable:
            token_ids = self.encode(text)

            for token_id in token_ids:
                yield token_id





    def decode(self,ids:list[int])->str:
        byte_sequences = []
        for token_id in ids:
            if token_id in self.vocab.keys():
                byte_sequences.append(self.vocab[token_id])


        combined_bytes = b''.join(byte_sequences)

        text = combined_bytes.decode('utf-8',errors='replace')

        return text

