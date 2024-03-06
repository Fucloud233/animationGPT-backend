from enum import Enum
from flask import Flask, request

from utils.translate import translate
from utils.hash import hash_string

app = Flask("animationGPT")

class LangKind(Enum):
    EN = 'en'
    CN = 'cn'

@app.route("/generate", methods=['POST'])
def generate():
    # 1. 输入 Prompt，选择生成语言
    try:
        prompt = request.json['prompt']
        lang = LangKind(request.json['language'])

        print(prompt, lang)
    except KeyError:
        return "key parameter not found"
    except ValueError:
        return "the kind of language not supported"

    # 3. 后端接收 Prompt，如果是中文，调用 API 翻译成英文
    if lang != LangKind.EN:
        (flag, prompt) = translate(prompt)
        if not flag:
            return "translate error"
        

    # 4. 通过 Hash 算法生成一个 id，将 id 和 prompt 保存到 Redis 中
    hash_code = hash_string(prompt)

    return hash_code
    

    # > Hash 算法和输入的语言有关，因此需要考虑语言分类

    # redis 注意保持会话 id 的问题

    return prompt


@app.route("/download", methods=['GET'])
def download():
    ...

if __name__ == '__main__':
    app.run(port=8081, host="0.0.0.0", debug = True)