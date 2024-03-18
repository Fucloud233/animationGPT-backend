import sys
import pandas as pd
from pprint import pprint
import json
import requests
from tqdm import tqdm

def main(file_path: str):
    with open(file_path, "r") as f:
        prompts = f.readlines()

    for prompt in tqdm(prompts):
        # 通过 requests 请求，生成
        requests.post("http://localhost:8082/generate", json = {
            "prompt": prompt,
            "language": "en"
        })


# covert json to csv
def parse(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
        for item in data:
            print(item)

if __name__ == '__main__':
    args = sys.argv

    file_path = 'data/examples.csv'
    if len(args) >= 2:
        file_path = args[1]

    main(file_path)
