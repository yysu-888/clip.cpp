import argparse
import os
import json

import numpy as np
from gguf import *
from transformers import CLIPModel
from transformers import ChineseCLIPProcessor, ChineseCLIPModel


def convert(parse_args):
    model =None
    if "chinese" in parse_args.model_dir.lower():
        model = ChineseCLIPModel.from_pretrained(parse_args.model_dir)
    elif "openai" in parse_args.model_dir:
        model = CLIPModel.from_pretrained(parse_args.model_dir)
    list_vars = model.state_dict()

    model_type = parse_args.convert_type

    outfile = os.path.join(parse_args.output_dir,"clip"+"_"+model_type+".gguf")
    gguf_writer = gguf.GGUFWriter(outfile, 'clip')

    print(f"Model tensors saved to {outfile}:")
    for k,v in list_vars.items():
        if "logit_scale" in k: continue  
        data = v.cpu().numpy()

        if "patch_embedding.weight" in k:
            data = data.astype(np.float16)

        if "chinese" in parse_args.model_dir and k.startswith("text_model"):
            k = k.replace("attention.self","self_attn").replace("query",
                    "q_proj").replace("key","k_proj").replace("value","v_proj").replace("attention.output.dense",
                    "self_attn.out_proj").replace("attention.output.LayerNorm","layn_0").replace("intermediate.dense","linear_0").replace(
                        "output.dense","linear_1").replace("output.LayerNorm","layn_1")

        if model_type =="f16":
            data = data.astype(np.float16)
        
        print(k,data.shape)
        gguf_writer.add_tensor(k, data)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

if __name__=="__main__":
    arg = argparse.ArgumentParser(prog="convert_hf_to_gguf.py")
    arg.add_argument("-model_dir",required=True, help="Path to model directory cloned from HF Hub")
    arg.add_argument("-convert_type", required=True, choices=['f32', 'f16'])
    arg.add_argument("-output_dir",default="./",help="Directory to save GGUF files. Default is the original model directory")

    parse_args = arg.parse_args()
    convert(parse_args)
