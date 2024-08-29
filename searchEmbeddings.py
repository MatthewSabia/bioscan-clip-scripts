import torch
import numpy as np
import h5py
import faiss
import json
import hydra
import pandas
import time
import random

def main():
    # get embeddings from file
    embeddings_file = h5py.File('extracted_features_of_all_keys.hdf5', 'r')

    # variable and index initialization
    dim = 768
    count = 0

    image_index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexFlatIP(dim)

    startTime = time.time()

    # create dict
    file_dict = {}
    ref_dict = {}
    i = 0
    for object in embeddings_file["file_name"]:
        str_obj = str(object)
        file_dict[str_obj] = embeddings_file["encoded_image_feature"][i:i+1]
        ref_dict[str_obj] = i
        i += 1
        # print(i)
    
    finishDict = time.time()
    dictTime = finishDict - startTime
    print("Time to make dict: ", dictTime)

    # print(file_list)

    # add embeddings to index
    for key in embeddings_file.keys():
        if (key == "encoded_image_feature"):
            embedding = embeddings_file[key][:]
            embedding = embedding.astype(np.float32)
            image_index.add(embedding)

    finishIndex = time.time()
    indexTime = finishIndex - finishDict
    print("Time to make index: ", indexTime)
    print()
    
    # b'4565393.jpg'
    while (True):
        raw_input_id = input("Enter a sample ID to search for: ")
        input_id = raw_input_id.lower()

        if (input_id == "exit"):
            exit()

        # get input in embedding form
        try:
            search_embedding = file_dict[raw_input_id]
            ID = True
        except:
            print("Given ID does not exist.")
            print()
            ID = False
        
        # search for embedding
        if (ID == True):
            num_neighbors = 10
            search_embedding = search_embedding.astype(np.float32)
            D, I = image_index.search(search_embedding, num_neighbors)
            print("The 10 closest embeddings are: ")
            for i in range(num_neighbors):
                print(str(i+1) + ".", I[0][i])
            print()
        
        

    
    



main()