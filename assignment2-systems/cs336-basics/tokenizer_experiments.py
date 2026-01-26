import enum
from cs336_basics import tokenizer
from typing import IO, Any, BinaryIO
from tests import test_tokenizer
import random
import numpy as np
import multiprocessing as mp
from functools import partial

TinyStories_Vocab_Path = './../TinyStories_Result/vocab.json'
TinyStories_Merges_Path = './../TinyStories_Result/merges.txt'

OpenWebText_Vocab_Path = './../OpenWebText_Result/vocab.json'
OpenWebText_Merges_Path = './../OpenWebText_Result/merges.txt'

TinyStories_Datapath = './../data/TinyStoriesV2-GPT4-train.txt'
OpenWebText_Datapath = './../data/owt_train.txt'

TinyStories_Valid_Datapath = './../data/TinyStoriesV2-GPT4-valid.txt'
OpenWebText_Valid_Datapath = './../data/owt_valid.txt'


def sample_documents_from_file(filepath,num_samples=10):
    documents = []

    with open(filepath,'r',encoding='utf-8') as f:
        content = f.read()

    parts = content.split('<|endoftext|>')

    for part in parts:
        if part.strip():
            documents.append(part+'<|endoftext|>')

    
    if len(documents) <= num_samples:
        return documents

    return random.sample(documents,num_samples)

def all_documents_from_file(filepath):
    documents = []

    with open(filepath,'r',encoding='utf-8') as f:
        content = f.read()

    parts = content.split('<|endoftext|>')

    for part in parts:
        if part.strip():
            documents.append(part+'<|endoftext|>')

    return documents

def calculate_compression_ratio(text,tokenizer):
    original_bytes = len(text.encode('utf-8'))

    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    compression_ratio = original_bytes / num_tokens if num_tokens > 0 else 0
    
    return compression_ratio

def encode_text(text,tokenizer):

    tokens = tokenizer.encode(text)

def encode_entire_file(filepath,tokenizer):

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 一次性编码整个文件内容
    tokens = tokenizer.encode(content)
    return tokens

def encode_documents_batch(docs_batch,tokenizer):
    tokens = []
    for doc in docs_batch:
        doc_tokens = tokenizer.encode(doc)
        tokens.extend(doc_tokens)
    return tokens

def encode_documents_parallel(documents, tokenizer, num_processes=None):

    if num_processes is None:
        num_processes = min(mp.cpu_count(), 96)  # 使用最多32个进程
    
    # 将文档分成批次
    batch_size = max(1, len(documents) // num_processes)
    doc_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    
    # 创建编码函数（固定tokenizer参数）
    encode_func = partial(encode_documents_batch, tokenizer=tokenizer)
    
    print(f"Using {len(doc_batches)} processes to encode {len(documents)} documents...")
    
    # 使用进程池并行处理
    with mp.Pool(processes=len(doc_batches)) as pool:
        results = pool.map(encode_func, doc_batches)
    
    # 合并所有批次的结果
    all_tokens = []
    for batch_tokens in results:
        all_tokens.extend(batch_tokens)
    
    return all_tokens

if __name__ == "__main__":
    
    print("Sampling documents...")
    tinystories_docs = sample_documents_from_file(TinyStories_Datapath, 10)
    openwebtext_docs = sample_documents_from_file(OpenWebText_Datapath, 10)

    
    print(f"Sampled {len(tinystories_docs)} TinyStories documents")
    print(f"Sampled {len(openwebtext_docs)} OpenWebText documents")


    print("\nLoading TinyStories tokenizer...")
    tinystories_tokenizer = test_tokenizer.get_tokenizer_from_vocab_merges_path(
        vocab_path=TinyStories_Vocab_Path,
        merges_path=TinyStories_Merges_Path,
        special_tokens=["<|endoftext|>"]
    )

    print("Loading OpenWebText tokenizer...")
    openwebtext_tokenizer = test_tokenizer.get_tokenizer_from_vocab_merges_path(
        vocab_path=OpenWebText_Vocab_Path,
        merges_path=OpenWebText_Merges_Path,
        special_tokens=["<|endoftext|>"]
    )


    print("\nCalculating TinyStories compression ratios...")
    tinystories_ratios = []
    for i, doc in enumerate(tinystories_docs):
        ratio = calculate_compression_ratio(doc, tinystories_tokenizer)
        tinystories_ratios.append(ratio)
        
    avg_tinystories_ratio = sum(tinystories_ratios) / len(tinystories_ratios)
    print(f"TinyStories compression ratios:{avg_tinystories_ratio}")


    print("\nCalculating OpenWebText compression ratios...")
    openwebtext_ratios = []
    for i, doc in enumerate(openwebtext_docs):
        ratio = calculate_compression_ratio(doc, openwebtext_tokenizer)
        openwebtext_ratios.append(ratio)
    
    avg_openwebtext_ratio = sum(openwebtext_ratios) / len(openwebtext_ratios)
    print(f"OpenWebText compression ratios:{avg_openwebtext_ratio}")


    print("\nCalculating OpenWebText Compression Ratio with TinyStories tokenizer...")
    openwebtext_ratios = []
    for i, doc in enumerate(openwebtext_docs):
        ratio = calculate_compression_ratio(doc, tinystories_tokenizer)
        openwebtext_ratios.append(ratio)
    
    avg_openwebtext_ratio = sum(openwebtext_ratios) / len(openwebtext_ratios)
    print(f"OpenWebText compression ratios:{avg_openwebtext_ratio}")

    # print("\nLoading all Dataset(Train/Valid)")

    # print("\nEncoding all TinyStories Dataset")

    # TinyStories_Valid_Encode = []
    # TinyStories_Valid_docs = all_documents_from_file(TinyStories_Valid_Datapath)
    # for i, doc in enumerate(TinyStories_Valid_docs):
    #     tokens = tinystories_tokenizer.encode(doc)
    #     TinyStories_Valid_Encode.extend(tokens)


    # TinyStories_Train_Encode = []
    # TinyStories_Train_docs = all_documents_from_file(TinyStories_Datapath)
    # for i,doc in enumerate(TinyStories_Train_docs):
    #     tokens = tinystories_tokenizer.encode(doc)
    #     TinyStories_Train_Encode.extend(tokens)




    # print("\nEncoding all OpenWebText Dataset")
    # OpenWebText_Train_Encode = []
    # OpenWebText_Train_docs = all_documents_from_file(OpenWebText_Datapath)
    # for i, doc in enumerate(OpenWebText_Train_docs):
    #     tokens = openwebtext_tokenizer.encode(doc)
    #     OpenWebText_Train_Encode.extend(tokens)
    
    # # 编码OpenWebText验证数据集
    # OpenWebText_Valid_Encode = []
    # OpenWebText_Valid_docs = all_documents_from_file(OpenWebText_Valid_Datapath)
    # for i, doc in enumerate(OpenWebText_Valid_docs):
    #     tokens = openwebtext_tokenizer.encode(doc)
    #     OpenWebText_Valid_Encode.extend(tokens)

    print("\nEncoding all TinyStories Dataset")
    
    print("Encoding TinyStories valid dataset...")
    TinyStories_Valid_docs = all_documents_from_file(TinyStories_Valid_Datapath)
    TinyStories_Valid_Encode = encode_documents_parallel(TinyStories_Valid_docs, tinystories_tokenizer)


    print("Encoding TinyStories train dataset...")
    TinyStories_Train_docs = all_documents_from_file(TinyStories_Datapath)
    TinyStories_Train_Encode = encode_documents_parallel(TinyStories_Train_docs, tinystories_tokenizer)
    

    print("\nEncoding all OpenWebText Dataset")
    

    print("Encoding OpenWebText valid dataset...")
    OpenWebText_Valid_docs = all_documents_from_file(OpenWebText_Valid_Datapath)
    OpenWebText_Valid_Encode = encode_documents_parallel(OpenWebText_Valid_docs, openwebtext_tokenizer)

    
    print("Encoding OpenWebText train dataset...")
    OpenWebText_Train_docs = all_documents_from_file(OpenWebText_Datapath)
    OpenWebText_Train_Encode = encode_documents_parallel(OpenWebText_Train_docs, openwebtext_tokenizer)
    


    print("\nSaving encoded datasets as uint16 NumPy arrays...")
    
    np.save('./../TinyStories_Result/train_tokens.npy', np.array(TinyStories_Train_Encode, dtype=np.uint16))
    np.save('./../TinyStories_Result/valid_tokens.npy', np.array(TinyStories_Valid_Encode, dtype=np.uint16))
    np.save('./../OpenWebText_Result/train_tokens.npy', np.array(OpenWebText_Train_Encode, dtype=np.uint16))
    np.save('./../OpenWebText_Result/valid_tokens.npy', np.array(OpenWebText_Valid_Encode, dtype=np.uint16))

    print("All datasets encoded and saved successfully!")
    print(f"TinyStories train tokens: {len(TinyStories_Train_Encode)}")
    print(f"TinyStories valid tokens: {len(TinyStories_Valid_Encode)}")
    print(f"OpenWebText train tokens: {len(OpenWebText_Train_Encode)}")
    print(f"OpenWebText valid tokens: {len(OpenWebText_Valid_Encode)}")

