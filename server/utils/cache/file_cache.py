from pathlib import Path
from random import sample
from typing import List

from threading import Thread

import pprint


'''
1. 在程序运行时，遍历缓存目录，读取存在的编号
2. 此时这些编号信息就保存在内存当中，程序判断是否已经生成不需要访问IO，会提升效率
3. 设置一个最大的次数，如果大于整个次数，则随机删除n%
4. 最后也不需要处理持久化的问题
'''


class DeleteThread(Thread):
    def __init__(self, cache_path: str, filenames: List[str]):
        Thread.__init__(self)
        self.filenames = filenames
        self.cache_path = Path(cache_path)
    
    def run(self):
        for filename in self.filenames:
            folder = Path.joinpath(self.cache_path, filename)
            for file in folder.iterdir():
                file.unlink
            folder.rmdir() 

class FileCache:
    def __init__(self, cache_path: str, max_count: int=2000, delete_rate: int=0.3):
        self.cache_path = Path(cache_path)
        self.max_count = max_count
        self.delete_num = int(max_count * delete_rate)
    
        self.data = set()

        self.__load()

    def add(self, id: str):
        if id in self.data:
            return False

        self.data.add(id)

        # 当创建的数量超过最大限制时，则开启新的线程随机删除文件
        if len(self.data) > self.max_count:
            self.__random_delete()

        return True

    def check(self, id: str) -> bool:
        return id in self.data

    def __load(self):
        self.data = set([ file.name for file in self.cache_path.iterdir() ])

    def __random_delete(self):
        # 删除内存中的集合同步处理
        ids = sample(list(self.data), self.delete_num)
        for id in ids:
            self.data.remove(id)

        # 删除文件系统的内容则使用异步
        thread = DeleteThread(self.cache_path, ids)
        thread.start()


         
        