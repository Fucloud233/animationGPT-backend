import requests
import time
import uuid
import json

from utils.hash import hash_string, HashKind
from translate_bot import TranslateBot

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

class YoudaoBot(TranslateBot):
    URL = 'https://openapi.youdao.com/api'

    def __init__(self, 
        app_key: str, app_secret: str, 
        lang_from: str="zh_CHS", lang_to: str="en"
    ):
        self.app_key = app_key
        self.app_secret = app_secret

        self.lang_from = lang_from
        self.lang_to = lang_to

    def check(self) -> bool:
        return self.translate("hello")[0]

    def translate(self, text: str):
        data = {
            'q': text,
            'from': self.lang_from,
            'to': self.lang_to
        }

        self.addAuthParams(data)

        header = {'Content-Type': 'application/x-www-form-urlencoded'}

        count = 3
        while(count > 0):
            res = requests.post(self.URL, data, header) 
            
            # 判断翻译 API 是否能够正常访问
            if res.status_code != 200:
                count -= 1
                continue
                
            content = json.loads(str( res.content, 'utf-8'))

            if content['errorCode'] != "0":
                break
            else:
                return (True, content['translation'][0])
            
        return (False, "")
            

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
    def addAuthParams(self, params):
        q = params.get('q')
        if q is None:
            q = params.get('img')
        q = "".join(q)
        salt = str(uuid.uuid1())
        curtime = str(int(time.time()))
        sign = calculateSign(self.app_key, self.app_secret, q, salt, curtime)
        params['appKey'] = self.app_key
        params['salt'] = salt
        params['curtime'] = curtime
        params['signType'] = 'v3'
        params['sign'] = sign
        
        

    