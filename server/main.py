from enum import Enum
from flask import Flask, request, send_file
from pathlib import Path
import logging

from utils.translate import get_bot as get_translate_bot
from utils.hash import hash_string
from utils.file import FileKind
from utils.npy2bvh import Joint2BVHConvertor

from utils.cache.file_cache import FileCache

from bot import T2MBot

class LangKind(Enum):
    EN = 'en'
    CN = 'cn'

VIDEO_TYPE = "video/mp4"
BVH_TYPE = "application/bvh"


app = Flask("animationGPT")
with app.app_context():
    bot = T2MBot()
    converter = Joint2BVHConvertor()

    logging.basicConfig(
        filename='results/animation.log', 
        format="%(asctime)s: [%(levelname)s] %(message)s ",
        level=logging.INFO
    ) 
    
    translate_bot = get_translate_bot()

    cache = FileCache("./cache", max_count=10)


@app.route("/generate", methods=['POST'])
def generate():
    # 1. 输入 Prompt，并选择输入的语言
    try:
        prompt = request.json['prompt']
        lang = LangKind(request.json['language'])

        # print(prompt, lang)
    except KeyError:
        return "key parameter not found", 400
    except ValueError:
        return "the kind of language not supported", 400

    # 3. 如果输入语言不为英文，则需要调用API翻译
    if lang != LangKind.EN:
        (flag, prompt) = translate_bot.translate(prompt)
        print("result: ", flag, prompt)
        if not flag:
            return "translate error", 500
        
    logging.info("cur prompt: " + prompt)
        
    # 4. 通过hash算法将Prompt转换称为16进制，并存储起来
    id = hash_string(prompt)
    
    if not cache.check(id):
        bot.generate_motion(prompt, id)
        cache.add(id)
    else:
        logging.info("cache hint!")

    # 5. 最后返回视频
    path = FileKind.MP4.to_cache_path(id)
    
    return send_file(path,download_name=id, mimetype=VIDEO_TYPE)


@app.route("/download", methods=['GET'])
def download():

    try:
        id = request.args['id']
    except KeyError:
        return "key parameter not found", 400

    if not cache.check(id):
        return "Session not found", 404
    
    npy_path = FileKind.NPY.to_cache_path(id)
    bvh_path = FileKind.BVH.to_cache_path(id)
    converter.convert(npy_path, bvh_path)

    return send_file(bvh_path, download_name=id, mimetype=BVH_TYPE)

if __name__ == '__main__':
    from waitress import serve
    serve(app, port=8082, host="0.0.0.0")