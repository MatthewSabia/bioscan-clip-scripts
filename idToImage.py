import torch
import numpy as np
import h5py
import faiss
import json
import hydra
import pandas
import time
import random
import io
import pickle
from PIL import Image

def get_image(file, dataset_image_mask, processid_to_index, query_id):
    idx = processid_to_index[query_id]
    image_enc_padded = file["image"][idx].astype(np.uint8)
    enc_length = dataset_image_mask[idx]
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image

def main():
    # get embeddings from file
    embeddings_file = h5py.File('5m/extracted_features_of_all_keys.hdf5', 'r')

    # variable and index initialization
    dim = 768
    count = 0
    startTime = time.time()

    # create dicts
    f = open("id_emb_dict.pickle", "rb")
    id_to_emb_dict = pickle.load(f)
    f = open("indx_to_id.pickle", "rb")
    indx_to_id_dict = pickle.load(f)
    dataset_hdf5_all_key = h5py.File('full5m/BIOSCAN_5M.hdf5', "r", libver="latest")['all_keys']
    dataset_processid_list = [item.decode("utf-8") for item in dataset_hdf5_all_key["processid"][:]]
    dataset_image_mask = dataset_hdf5_all_key["image_mask"][:]
    processid_to_index = {pid: idx for idx, pid in enumerate(dataset_processid_list)}

    # i = 0
    # for object in dataset_processid_list:
    #     id_to_emb_dict[object] = embeddings_file["encoded_image_feature"][i:i+1]
    #     i += 1
    #     if (i < 100):
    #         print(object)
    
    finishDict = time.time()
    dictTime = finishDict - startTime
    print("Time to make dict: ", dictTime)

    image_index = faiss.read_index("image_index.index")

    # # add embeddings to index
    # image_index = faiss.IndexFlatIP(dim)
    # for key in embeddings_file.keys():
    #     if (key == "encoded_image_feature"):
    #         embedding = embeddings_file[key][:]
    #         embedding = embedding.astype(np.float32)
    #         image_index.add(embedding)

    # finishIndex = time.time()
    # indexTime = finishIndex - finishDict
    # print("Time to make index: ", indexTime)
    # print()
    # faiss.write_index(image_index, "image_index.index")

    # queryID = 'CRPEB22183-21'


    # b'4565393.jpg'
    while (True):
        raw_input_id = input("Enter a sample ID to search for: ")
        input_id = raw_input_id.lower()

        if (input_id == "exit"):
            exit()

        if (input_id == 'stop'):
            break

        # get input in embedding form
        try:
            query = id_to_emb_dict[raw_input_id]
            ID = True
        except:
            print("Given ID does not exist.")
            print()
            ID = False
        
        # search for embedding
        if (ID == True):
            num_neighbors = 10
            query = query.astype(np.float32)
            D, I = image_index.search(query, num_neighbors)
            print("The 10 closest embeddings are: ")
            for i in range(num_neighbors):
                indx = int(I[0][i])
                print(str(i+1) + ".", indx)

                queryID = indx_to_id_dict[indx]
                print(queryID)
                image = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, queryID)

                image.thumbnail((128, 128))
                fname = "im" + str(i) + ".jpg"
                image.save(fname + ".thumbnail", "JPEG")

            print()
        
        
            


            
            dataset_image_mask = dataset_hdf5_all_key["image_mask"][:]
            processid_to_index = {pid: idx for idx, pid in enumerate(dataset_processid_list)}

            image = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, queryID)
            print("show1")
            image.show()
            image.thumbnail((128, 128))
            image.save("im.jpg" + ".thumbnail", "JPEG")
            print("show2")

    # 'ARONZ671-20'
    

main()