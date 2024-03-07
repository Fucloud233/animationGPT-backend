from enum import Enum
from flask import Flask, request, send_from_directory, send_file
from pathlib import Path

from utils.translate import translate
from utils.hash import hash_string
from store import Store
from bot import T2MBot

app = Flask("animationGPT")
bot = T2MBot()

class LangKind(Enum):
    EN = 'en'
    CN = 'cn'

@app.route("/generate", methods=['POST'])
def generate():
    # 1. 输入 Prompt，并选择输入的语言
    try:
        prompt = request.json['prompt']
        lang = LangKind(request.json['language'])

        # print(prompt, lang)
    except KeyError:
        return "key parameter not found"
    except ValueError:
        return "the kind of language not supported"

    # 3. 如果输入语言不为英文，则需要调用API翻译
    if lang != LangKind.EN:
        (flag, prompt) = translate(prompt)
        if not flag:
            return "translate error"
        

    # 4. 通过hash算法将Prompt转换称为16进制，并存储起来
    id = hash_string(prompt)
    
    if not Store.check_exist(id):
        Store.set(id, prompt)
        bot.generate_motion(prompt, id)

    # 5. 最后返回视频
    path = Path.joinpath(Path("cache"), id, "video.mp4")
    
    return send_file(path)


@app.route("/download", methods=['GET'])
def download():
    ...

if __name__ == '__main__':
    app.run(port=8081, host="0.0.0.0", debug = True)