'''
{
    id: {
        prompt: "",
        mp4: "",
        bgh: "",
        # 如果没有Download，则不会删除
        hasDownload: boolean
    }
}

'''

import redis

HOST = "localhost"
PORT = 6379

UTF8 = "utf-8"

PROMPT = "prompt"
DOWNLOADED = "downloaded"

class Field:
    def __init__(self, name: str, type: type):
        self.name = name
        self.type = type


def new_info(prompt: str) -> dict:
    return {
        PROMPT: prompt,
        DOWNLOADED: 0
    }

    
class Store:
    r = redis.Redis(host=HOST, port=PORT, db=0)
    
    fields = [Field(PROMPT, str), Field(DOWNLOADED, bool)]

    def __get_field(id: str, field: Field):
        result = Store.r.hget(id, field.name)
        
        if result is None:
            return None
        
        result = result.decode(UTF8)
        if field.type is bool:
            return int(result) != 0
        else:
            return field.type(result)

    @staticmethod
    def get(id: str):
        result = {}
        for field in Store.fields:
            result[field.name] = Store.__get_field(id, field)

        return result
    
    @staticmethod
    def check_exist(id: str) -> bool:
        return Store.r.hgetall(id) is {}

    @staticmethod
    def set(id: str, prompt: str):
        Store.r.hset(id, mapping=new_info(prompt))


if __name__ == "__main__":
    Store.set("1234", "hello")
    print(Store.get("1234"))


    
