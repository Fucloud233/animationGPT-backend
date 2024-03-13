from pathlib import Path
from enum import Enum

BASE_PATH = "./cache"

class FileKind(Enum):
    NPY = "joints.npy"
    BVH = "data.bvh"
    GIF = "video.gif"
    MP4 = "video.mp4"

    def to_cache_path(self, id):
        return Path.joinpath(Path(BASE_PATH), str(id), self.value)
        


if __name__ == "__main__":
    kind = FileKind.NPY
    print(kind.to_cache_path("123"))
    print(FileKind.to_cache_path(kind, "123"))