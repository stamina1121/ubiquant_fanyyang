import datasets
import pprint

data_from_ar1 = datasets.load_dataset('parquet', data_files='./data_old/lean/train.parquet', split='train')
data_from_orm = datasets.load_dataset('parquet', data_files='/AI4M/users/ytwang/auto-proof/Agent-R1-Lean/data/EI_proofnet_mathlib_aug_shuffle_fewshot_prompt/8689_shuffle_prompt_train.parquet', split='train')

pprint.pp(data_from_ar1[0])
pprint.pp(data_from_orm[0])