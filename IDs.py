import gradio as gr
import torch
import numpy as np
import h5py
import faiss
import json
import random




def main():
    print("main")
    dataset_hdf5_all_key = h5py.File('full5m/BIOSCAN_5M.hdf5', "r", libver="latest")['all_keys']
    dataset_processid_list = [item.decode("utf-8") for item in dataset_hdf5_all_key["processid"][:]]
    processid_to_index = {pid: idx for idx, pid in enumerate(dataset_processid_list)}

    print("pre loop")
    i = 0
    for object in dataset_processid_list:
        if (i > 162830 and i < 162840):
            print(i, object)
        if (object == "AACTA1003-20"):
            print(i, object)
            
        i += 1
    print("in loop")



main()