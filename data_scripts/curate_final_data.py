from data.prompts import system_prompt, user_prompt
import pandas as pd
import datasets
import json
from copy import deepcopy

def load_data():
    with open("data/bird-verified-train.json") as f:
        train_data = json.load(f)

    with open("data/bird-verified-val.json") as f:
        val = json.load(f)

    with open("data/arcwise_plat_full.json") as f:
        test_full = json.load(f)

    with open("data/arcwise_plat_sql.json") as f:
        test_sql = json.load(f)

    val_test_data = []

    for data in test_full:
        data["split"] = "arcwise-plat-full"
        val_test_data.append(data)

    for data in test_sql:
        data["split"] = "arcwise-plat-sql"
        val_test_data.append(data)

    for data in val:
        data["split"] = "validation"
        val_test_data.append(data)

    return pd.DataFrame(train_data), pd.DataFrame(val_test_data)

def convert_data(data: pd.DataFrame):
    """
    format of the output dataframe:
    - data_source: "clean", "noisy", or "unknown
    - prompt: []
    - question_id: int
    -
    """
    converted_data = []
    for i, d in data.iterrows():
        db_name = d['db_id']
        with open(f"data/ddls/{db_name}.sql", "r", encoding="cp1252") as f:
            schema = f.read()
        converted_d = {
            'data_source': d['split'],
            'prompt': [
                {
                    'content': system_prompt,
                    'role': 'system'
                },
                {
                    'content': user_prompt.format(
                        schema=schema,
                        external_knowledge=d['evidence'],
                        question=d['question'],),
                    'role': 'user'
                }
            ],
            'env_class': 'text2sql',
            'db_id': d['db_id'],
            'external_knowledge': d['evidence'],
            'data': 'bird-verified',
            'reward_spec': {
                'ground_truth': d['SQL'],
                'method': d['grading_method'] if 'grading_method' in d else 'set'
            },
            'extra_info': {'index': 0, 'split': 'dummy'},
            'question_id': str(d['question_id'])
        }
        converted_data.append(converted_d)
    return pd.DataFrame(converted_data)

def curate_train_val_test_dataset():
    train_data, val_test_data = load_data()
    train_data["split"] = "train"
    train_data = convert_data(train_data)
    val_test_data = convert_data(val_test_data)

    train_data.to_json("data/inspect_train.json", orient="records", indent=2)
    val_test_data.to_json("data/inspect_val_test.json", orient="records", indent=2)
    train_dataset = datasets.Dataset.from_pandas(train_data)
    val_test_dataset = datasets.Dataset.from_pandas(val_test_data)

    train_dataset.to_parquet("data/bird-verified-train.parquet")
    val_test_dataset.to_parquet("data/val_test.parquet")


if __name__ == "__main__":
    curate_train_val_test_dataset()
