import hashlib
from enum import Enum

'''
不同哈希算法的区别： 
* 校验长度：md5(16), sha1(20), sha256(32)
* 运行速度：md5 > sha1 > sha256

由于本项目不涉及加密需求，因此选择运行效率最快即可，
16 Bytes 也完全够日常使用。
'''

class HashKind(Enum):
    MD5 = "md5"
    SHA1 = "SHA1"
    SHA256 = "sha256"


def hash_string(text: str, kind: HashKind=HashKind.MD5):
    bytes = text.encode('utf-8')
    
    if (kind == HashKind.SHA1):
        hash_object = hashlib.sha1()
    elif(kind == HashKind.SHA256):
        hash_object = hashlib.sha256()
    else:
        hash_object = hashlib.md5()
    
    hash_object.update(bytes)
        
    return hash_object.hexdigest() 