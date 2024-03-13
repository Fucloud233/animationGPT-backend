from cache import Cache, Node, MAX_NUM
from simple_cache import SimpleCache

# 节点的值可能会对影响刷新情况
# Node {
#   value: ..., isDownloaded: ...
# }

'''
    客户端对服务端的访问情况
    key -> ssid value -> prompt
    1. generate：向缓冲中添加对应的键值对，对于大多数情况下，会加入inactive队列中
    2. download：下载并不会影响这个缓存的刷新情况。
    '''

class PlusCache(Cache):
    def __init__(self, 
        delete_call_back, 
        max_num: int=MAX_NUM, 
        active_rate: float=0.7
    ):
        self.active_max_num = active_rate * max_num
        self.inactive_max_num = max_num - active_rate

        self.active_cache = SimpleCache(None, self.active_max_num)
        self.inactive_cache = SimpleCache(delete_call_back, self.inactive_max_num)

    def add(self, key, value):
        # # 如何活跃链表存在，这需要直接刷新到最前，然后将删除的节点放在非活跃情况
        # if self.active_cache.check(key):
        #     deleted_node = self.active_cache.refresh_by_key(key)
        #     if deleted_node != None:
        #         self.inactive_cache.add(deleted_node)
        # # 如果在非活跃链表存在，则需要刷新到活跃链表中最前，然后直接删除最后的
        # elif self.inactive_cache.check(key):
        #     self.inactive_cache.remove_by_key(key)
        #     self.active_cache.add(key, value)

        if self.check(key):
            raise ValueError("Duplicated Key")
        
        self.inactive_cache.add(key, value)    
        return True
            
    def check(self, key):
        return self.active_cache.check(key) or self.inactive_cache.check(key)

    def get(self, key):
        node: Node = None
        if self.active_cache.check(key):
            node = self.active_cache.get(key)
        # 如果在非活跃链表中访问，则会申请到活跃链表中
        elif self.inactive_cache.check(key):
            node = self.inactive_cache.remove_by_key(key)
            self.active_cache.add_node(node)

        return node.value
    
    # 主要更新状态
    def update(self, key, value):
        if self.active_cache.check(key):
            self.active_cache.update(key, value, False)
        elif self.inactive_cache.check(key):
            self.inactive_cache.update(key, value, False)

    def display(self):
        self.active_cache.display()
        self.inactive_cache.display()
        
            
if __name__ == "__main__":
    cache = PlusCache(None, 5)

    class Value:
        def __init__(self, value, method):
            self.value = value
            self.method = method


    # 其中 method：0 代表添加，1 代表获取
    data = [
        Value(3, 0), Value(4, 0), Value(5, 0), Value(6, 0), Value(5, 1), Value(6, 1)
    ]

    for value in data:
        if value.method == 0:
            cache.add(value.value, value.value)
        elif value.method == 1:
            cache.get(value.value)

        cache.display()
        print("------")
    