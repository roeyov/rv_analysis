from only_mcmc_from_model_single import run_mcmc
from compare_best_row_selectors import get_best_row, BestRowConfig
import json

from tmps.reorganize_solutions import out_dir

json_param_file = '/Users/roeyovadia/Roey/Masters/Reasearch/Scripts/params.json'  # contains ONLY args_dict
with open(json_param_file, "r") as f:
    args_dict = json.load(f)


def get_row_by_solution_id(df,sol_id):
    return df.loc[df['solution_id'] == sol_id].iloc[0]

def run_mcmc_on_solution(df,sol_id):
    row = get_row_by_solution_id(df,sol_id)
    run_mcmc(args_dict=json_param_file,
             best_row=row,
             MJDs=mjds,
             rv_obs=rv_obs,
             rv_sigmas=rv_sigmas,
             out_dir=out_dir,
             star_name=star_name)

def run_mcmc_on_best_row(df,config):
    row = get_best_row(df, config)
    run_mcmc(args_dict=json_param_file,
             best_row=row,
             MJDs=mjds,
             rv_obs=rv_obs,
             rv_sigmas=rv_sigmas,
             out_dir=out_dir,
             star_name=star_name)