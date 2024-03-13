import sys; sys.path.append("./server")

from utils.cache.file_cache import FileCache
from pathlib import Path
import random as ra

def main():
    path = Path("./results/test")
    if not path.exists():
        path.mkdir()

    cache = FileCache(path, max_count=10)
    
    for i in range(20):
        id = ra.randint(100000, 999999)
        folder_path = Path.joinpath(path, str(id))
        folder_path.mkdir()

        cache.add(str(id))

    # for file in path.iterdir():
    #     file.rmdir()

    


if __name__ == '__main__':
    main()
