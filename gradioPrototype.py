import gradio as gr
import torch
import numpy as np
import h5py
import faiss
from PIL import Image
import io 
import pickle

def get_image(image1, image2, dataset_image_mask, processid_to_index, idx):
    # idx = processid_to_index[query_id]
    if (idx < 162834):
        image_enc_padded = image1[idx].astype(np.uint8)
    elif(idx >= 162834):
        image_enc_padded = image2[idx-162834].astype(np.uint8)

    enc_length = dataset_image_mask[idx]
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image

def searchEmbeddings(id):
    # variable and index initialization
    dim = 768
    count = 0
    num_neighbors = 10

    image_index = faiss.IndexFlatIP(dim)

    # load dictionaries
    # with open("id_emb_dict.pickle", "rb") as f:
    #     id_to_emb_dict = pickle.load(f)
    # with open("indx_to_id.pickle", "rb") as f:
    #     indx_to_id_dict = pickle.load(f)

    # get index
    image_index = faiss.read_index("image_index.index")

    # search for query
    query = id_to_emb_dict[id]
    query = query.astype(np.float32)
    D, I = image_index.search(query, num_neighbors)

    id_list = []
    i = 1
    for indx in I[0]:
        id = indx_to_id_dict[indx]
        id_list.append(id)

    image1 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][0])
    image2 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][1])
    image3 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][2])
    image4 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][3])
    image5 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][4])
    image6 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][5])
    image7 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][6])
    image8 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][7])
    image9 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][8])
    image10 = get_image(dataset_image1, dataset_image2, dataset_image_mask, processid_to_index, I[0][9])

    # return id_list, id_list[0], id_list[1], id_list[2], id_list[3], id_list[4], id_list[5], id_list[6], id_list[7], id_list[8], id_list[9], image1, image2, image3, image4, image5, image6, image7, image8, image9, image10
    # return id_list, indx_to_id_dict[I[0][0]], indx_to_id_dict[I[0][1]], indx_to_id_dict[I[0][2]], indx_to_id_dict[I[0][3]], indx_to_id_dict[I[0][4]], indx_to_id_dict[I[0][5]], indx_to_id_dict[I[0][6]], indx_to_id_dict[I[0][7]], indx_to_id_dict[I[0][8]], indx_to_id_dict[I[0][9]]
    return id_list, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10

with gr.Blocks() as demo:

    with open("dataset_processid_list.pickle", "rb") as f:
        dataset_processid_list = pickle.load(f)
    with open("dataset_image_mask.pickle", "rb") as f:
        dataset_image_mask = pickle.load(f)
    with open("processid_to_index.pickle", "rb") as f:
        processid_to_index = pickle.load(f)
    with open("dataset_image1.pickle", "rb") as f:
        dataset_image1 = pickle.load(f)
    with open("dataset_image2.pickle", "rb") as f:
        dataset_image2 = pickle.load(f)
    with open("id_emb_dict.pickle", "rb") as f:
        id_to_emb_dict = pickle.load(f)
    with open("indx_to_id.pickle", "rb") as f:
        indx_to_id_dict = pickle.load(f)

    with gr.Column():
        process_id = gr.Textbox(label="ID:", info="Enter a sample ID to search for")
        process_id_list = gr.Textbox(label="Closest 10 matches:" )
        search_btn = gr.Button("Search") 

    with gr.Row():
        image1 = gr.Image(label=1)
        image2 = gr.Image(label=2)
        image3 = gr.Image(label=3)
        image4 = gr.Image(label=4)
        image5 = gr.Image(label=5)
    with gr.Row():
        image6 = gr.Image(label=6)
        image7 = gr.Image(label=7)
        image8 = gr.Image(label=8)
        image9 = gr.Image(label=9)
        image10 = gr.Image(label=10)
    
    search_btn.click(fn=searchEmbeddings, inputs=process_id, 
                     outputs=[process_id_list, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10])

# ARONZ671-20
demo.launch()