import os
import pandas as pd
import numpy as np
import argparse
import tqdm
import cProfile
from utils.utils import make_runlog
from Models import ClassicModel, PyTorchModel, MultiOutputNN
from concurrent.futures import ThreadPoolExecutor, as_completed

# ClassicModel Parameters
RV_DIFF = 20.0
SNR_SIGNIFICANCE = 4.0

# PyTorchModel Parameters
# PYTORCH_MODEL = someModel
base_dir = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\torchModelsLocal\train_data_2_2000000\{}.{}"
MODEL_DIR = base_dir.format("model","pth")
# SCALER_DIR = base_dir.format("scaler","pickle")
classic_model = ClassicModel(RV_DIFF, SNR_SIGNIFICANCE)
# torch_model = PyTorchModel(MODEL_DIR,SCALER_DIR)
MODELS = {classic_model.__class__.__name__: classic_model,
          # torch_model.__class__.__name__: torch_model,
          }


def process_file(input_path, output_path, model, par_file):
    fp = os.path.join(input_path, par_file)
    df = pd.read_parquet(fp)
    pre_df_cols = df.columns
    model.predict(df)
    post_df_cols = df.columns
    unique_cols = [col for col in post_df_cols if col not in pre_df_cols]
    df[unique_cols].to_parquet(os.path.join(output_path, par_file))


def calc_model(input_path, output_path, model, max_workers=50):
    files = os.listdir(input_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, input_path, output_path, model, par_file)
            for par_file in files
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()  # This will raise any exceptions that occurred during execution


# def calc_model(input_path, output_path, model):
#     files = os.listdir(input_path)
#     for par_file in tqdm.tqdm(files):
#         fp = os.path.join(input_path, par_file)
#         df = pd.read_parquet(fp)
#         pre_df_cols = df.columns
#         model.predict(df)
#         post_df_cols = df.columns
#         unique_cols = [col for col in post_df_cols if col not in pre_df_cols]
#         df[unique_cols].to_parquet(os.path.join(output_path, par_file))


def main():
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add arguments
    parser.add_argument('--model', type=str, help='The wanted model',choices=MODELS.keys(),required=True)
    parser.add_argument('--output_dir', type=str, help='The output dir',required=True)
    parser.add_argument('--input_dir', type=str, help='The input dir',required=True)

    # Parse the arguments
    args = parser.parse_args()
    if os.path.isdir(args.output_dir):
        if len(os.listdir(args.output_dir)):
            print("outDir {} is not empty. choose different out".format(args.output_dir))
            exit(1)
    os.makedirs(os.path.join(args.output_dir,"model_output"))
    make_runlog(args.output_dir)

    model = MODELS[args.model]

    calc_model(args.input_dir, os.path.join(args.output_dir,"model_output"), model)

if __name__ == '__main__':
    # cProfile.run('main()', sort='cumulative')
    main()
