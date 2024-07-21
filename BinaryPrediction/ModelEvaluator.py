import os
import pandas as pd
import numpy as np
import argparse

from Models import ClassicModel, PyTorchModel

# ClassicModel Parameters
RV_DIFF = 20.0
SNR_SIGNIFICANCE = 4.0

# PyTorchModel Parameters
PYTORCH_MODEL = someModel
MODEL_DIR = "path/to/model"
MODELS = {str(type(ClassicModel)): ClassicModel(RV_DIFF, SNR_SIGNIFICANCE),
          str(type(PyTorchModel)): PyTorchModel(PYTORCH_MODEL, MODEL_DIR)
          }

def load_input_from_parquet(input_path):
    dfs = []
    files = os.listdir(input_path)
    for par_file in files:
        fp = os.path.join(input_path, par_file)
        dfs.append(pd.read_parquet(fp))
    return pd.concat(dfs, axis=0)

def main():
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add arguments
    parser.add_argument('--model', type=str, help='The input file',choices=MODELS.keys())
    parser.add_argument('--output_dir', type=str, help='The output dir',)
    parser.add_argument('--input_dir', type=str, help='The input dir',)

    # Parse the arguments
    args = parser.parse_args()
    if (os.path.exists(args.output)):
        print("The output folder already exist. delete it or choose new path")

    model = MODELS[args.model]
    df = load_input_from_parquet(args.input)
    out_dict = model.prediction_distribution(df)


if __name__ == '__main__':
    main()
