import torch
import numpy as np
import h5py
import faiss
import json
import hydra
import pandas
import time
import random

def flatL2(file, d, features):
    print("Testing FlatL2 Index")
    startTime = time.time()

    index = faiss.IndexFlatL2(d)
    for key in file.keys():
        if (key == "encoded_image_feature"):
            embedding = file[key][:]
            embedding = embedding.astype(np.float32)
            index.add(embedding)

    indexTime = time.time() - startTime
    startSearchTime = time.time()

    # search for 1000 random vectors
    num_searches = 0
    while (num_searches < 1000):
        data = random.choice(features)
        rnum = random.randrange(0, embedding.shape[1] - 1)
        query = file[data][rnum:(rnum + 1)]
        query = query.astype(np.float32)
        D, I = index.search(query, 5)
        num_searches += 1

    timePerQuery = (time.time() - startSearchTime) / 1000

    print("Index build time:", indexTime, "seconds")
    print("Time per query:", timePerQuery, "seconds")
    print()
        


def flatIP(file, d, features):
    print("Testing FlatIP Index")
    startTime = time.time()

    index = faiss.IndexFlatIP(d)
    for key in file.keys():
        if (key == "encoded_image_feature"):
            embedding = file[key][:]
            embedding = embedding.astype(np.float32)
            index.add(embedding)

    indexTime = time.time() - startTime
    startSearchTime = time.time()

    # search for 1000 random vectors
    num_searches = 0
    while (num_searches < 1000):
        data = random.choice(features)
        rnum = random.randrange(0, embedding.shape[1] - 1)
        query = file[data][rnum:(rnum + 1)]
        query = query.astype(np.float32)
        D, I = index.search(query, 5)
        num_searches += 1

    timePerQuery = (time.time() - startSearchTime) / 1000

    print("Index build time:", indexTime, "seconds")
    print("Time per query:", timePerQuery, "seconds")
    print()

    
def flatIVF(file, d, features):
    print("Testing IVFFlat Index")

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 128)

    print("Index build time:", indexTime, "seconds")
    print("Time per query:", timePerQuery, "seconds")
    print()


def main():
    embeddings_file = h5py.File('extracted_features_of_all_keys.hdf5', 'r')
    dim = 768

    types_of_features = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    ]

    flatL2(embeddings_file, dim, types_of_features)
    flatIP(embeddings_file, dim, types_of_features)
    flatIVF(embeddings_file, dim, types_of_features)



main()