import torch
import numpy as np
import h5py
import faiss
import json
import hydra
import pandas
import time

def index_data(name, obj):
    # embeddings.visititems(index_data)
    # print(name)
    print(obj)

    if isinstance(obj, h5py.Dataset):
        data = obj[()][:]
        # print(data)
        # return data


def main():

    # get embeddings from file and initialize index
    embeddings_file = h5py.File('extracted_feature_from_val_split.hdf5', 'r')
    dim = 768
    index = faiss.IndexFlatIP(dim)

    startTime = time.time()
    count = 0
    # add all necessary embeddings to index
    for key in embeddings_file.keys():
        keyGet = embeddings_file[key]
        for data in keyGet:
            if (data != "concatenated_feautre" and data != "averaged_feature" and embeddings_file[key][data][:].shape[1] == dim):
                embedding = embeddings_file[key][data][:]
                embedding = torch.from_numpy(embedding).cuda()
                embedding = torch.nn.functional.normalize(embedding, dim=1)
                embedding = embedding.cpu().numpy()
                index.add(embedding)
                count += 1
                # index.add(embeddings_file[key][data][:])
            
    # print total time elapsed
    endTime = time.time()
    totalTime = endTime - startTime
    print("Time to build index: ", totalTime)
    print("Count: ", count)
    # print(totalTime)



    # embeddings = np.array(embeddings, dtype=np.float32)
    # embeddings = embeddings.astype(np.float32)

    # # might need to normalize data? unsure
    # embeddings = torch.from_numpy(embeddings).cuda() # creates tensor from numpy array
    # embeddings = torch.nn.functional.normalize(embeddings) # normalize tensor
    # embeddings = embeddings.cpu().numpy() # convert tensors back to numpy array

    # # create index and add datasets to the index
    # d = 2 # 2 was dimension size when printing vectors in visititems
    # index = faiss.IndexFlatIP(d) 
    # index.add(embeddings)

    # print(index.is_trained)
    # print(index.ntotal)

    # query = input("Enter query: ")
    # numbers = input("Enter number of neighbors: ")

    # # record start time
    # startTime = time.time()

    # # search and print results
    # D, I = index.search(query, numbers)
    # print("The closest " + numbers + " models are:\n")
    # for i in numbers:
    #     print(I[:i])


    # # print total time elapsed
    # endTime = time.time()
    # totalTime = startTime - endTime
    # print("Time: " + totalTime)




main()