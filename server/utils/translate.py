import requests
import time
import uuid
import json

from pprint import pprint

from utils.hash import hash_string, HashKind

def load_app_key(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)

        return (config['appKey'], config['appSecret'])


'''
添加鉴权相关参数 -
    appKey : 应用ID
    salt : 随机值
    curtime : 当前时间戳(秒)
    signType : 签名版本
    sign : 请求签名
    
    @param appKey    您的应用ID
    @param appSecret 您的应用密钥
    @param paramsMap 请求参数表
'''
def addAuthParams(params, config_path):
    (appKey, appSecret) = load_app_key(config_path)

    q = params.get('q')
    if q is None:
        q = params.get('img')
    q = "".join(q)
    salt = str(uuid.uuid1())
    curtime = str(int(time.time()))
    sign = calculateSign(appKey, appSecret, q, salt, curtime)
    params['appKey'] = appKey
    params['salt'] = salt
    params['curtime'] = curtime
    params['signType'] = 'v3'
    params['sign'] = sign


def returnAuthMap(appKey, appSecret, q):
    salt = str(uuid.uuid1())
    curtime = str(int(time.time()))
    sign = calculateSign(appKey, appSecret, q, salt, curtime)
    params = {'appKey': appKey,
              'salt': salt,
              'curtime': curtime,
              'signType': 'v3',
              'sign': sign}
    return params


'''
    计算鉴权签名 -
    计算方式 : sign = sha256(appKey + input(q) + salt + curtime + appSecret)
    @param appKey    您的应用ID
    @param appSecret 您的应用密钥
    @param q         请求内容
    @param salt      随机值
    @param curtime   当前时间戳(秒)
    @return 鉴权签名sign
'''
def calculateSign(appKey, appSecret, q, salt, curtime):
    strSrc = appKey + getInput(q) + salt + curtime + appSecret
    return hash_string(strSrc, kind=HashKind.SHA256)

def getInput(input):
    if input is None:
        return input
    inputLen = len(input)
    return input if inputLen <= 20 else input[0:10] + str(inputLen) + input[inputLen - 10:inputLen]

def doCall(url, header, params, method):
    if 'get' == method:
        return requests.get(url, params)
    elif 'post' == method:
        return requests.post(url, params, header)


URL = 'https://openapi.youdao.com/api'

CONFIG_PATH = "configs/translate.json"


def translate(text: str, lang_from: str="zh_CHS", lang_to: str="en" ):
    # 1. 封装请求数据
    data = {'q': text, 'from': lang_from, 'to': lang_to }
    # 该函数会从配置文件中自动获取API_KEY
    addAuthParams(data, CONFIG_PATH)
    header = {'Content-Type': 'application/x-www-form-urlencoded'}

    # 2. 发送请求
    res = doCall(URL, header, data, 'post')

    # 3. 处理请求结果
    content = json.loads(str( res.content, 'utf-8'))
    # pprint(content)
    if content['errorCode'] != "0":
        return (False, "")
    else:
        return (True, content['translation'][0])
    

if __name__ == "__main__":
    input = "角色双手自然下垂于身体两侧，身体重心稍微下压，向左缓慢移动。"
    print(translate(input))