from cache import Cache,Node, MAX_NUM


'''
    需要实现的功能
    1. 添加新的数据
    2. 判断是否存在
    3. 当添加过多了，则需要考虑删除旧的
'''

class SimpleCache(Cache):
    def __init__(self, delete_call_back, max_num: int=MAX_NUM):
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.pre = self.head

        self.map = {}

        self.delete_call_back = delete_call_back
        self.max_num = max_num

    def remove_by_key(self, key):
        node = self.map.get(key)
        if node == None:
            return None
        
        return self.remove(node)
        
    def remove(self, node: Node):
        node.pre.next = node.next
        node.next.pre = node.pre

        return node

    def add(self, key, value):
        if self.map.get(key) is not None:
            return False

        node = Node(key, value)
        self.map[key] = node

        return self.__refresh(node); 
    
    def add_node(self, node: Node):
        self.map[node.key] = node

        return self.__refresh(node)
        

    # check 只负责状态查询，不会对修改结构中的属性
    def check(self, key):
        return self.map.get(key) != None
    
    def get(self, key):
        node = self.map.get(key)

        if node != None:
            self.__refresh(node)
            return node.value
        else:
            return None

    def update(self, key, value, need_refresh: bool):
        node: Node = self.map.get(key)
        if node is None:
            raise KeyError
        
        node.value = value
        
        if node.pre == self.head:
            return
        
        # 首先从该队列中删除该元素
        node.pre.next = node.next

        if need_refresh:
            self.__refresh(node)

    def __remove_last_used(self):
        node: Node = self.tail.pre
        if node == None:
            return

        self.map.pop(node.key)
        node.pre.next = self.tail
        self.tail.pre = node.pre
        
        if self.delete_call_back != None:
            self.delete_call_back(node.key, node.value)

        return node

    def refresh_by_key(self, key):
        node = self.map.get(key)
        return node != None if self.__refresh(node) else None

    def __refresh(self, node: Node):
        if node.pre != self.head:
            # 修改node节点的指针
            node.pre = self.head
            node.next = self.head.next

            # 将node插入 head 后面
            self.head.next.pre = node
            self.head.next = node

        if len(self.map) > self.max_num:
            deleted_node = self.__remove_last_used()

            return deleted_node
        
        return None

    def display(self):
        node = self.head.next
        while node != self.tail:
            print(f"({node.key}, {node.value})", end=" ")
            node = node.next
        print()
            
        
if __name__ == "__main__":
    def callback(key, value):
        print(key, value)

    cache = Cache(callback, 5)

    data = [1, 2, 4, 5, 6, 7, 8, 9]

    for d in data:
        cache.add(d, d)

    


        
        
        
        
        
        
        

        
        
        
    