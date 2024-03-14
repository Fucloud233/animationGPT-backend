import requests
from datetime import datetime

import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    df = pd.read_csv("./server/test/introduce.csv")
    prompts = df.get("text_en").to_list()

    times = []
    # for prompt in tqdm(prompts[0:10]):
    for prompt in prompts[0:2]:
        start_t = datetime.now()


        requests.post("http://localhost:8082/generate", json = {
            "prompt": prompt,
            "language": "en"
        })

        delta_t = (datetime.now() - start_t)

        times.append(delta_t.seconds +  delta_t.microseconds/ 1e6)
    

    print("result:", sum(times)/ len(times))
    # print(prompt)

    