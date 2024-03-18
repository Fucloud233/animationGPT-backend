import sys; sys.path.append("./server/utils/translate")
import json
from enum import Enum

config_path = "./configs/translate.json"

def read_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    
class TranslateBotKind(Enum):
    Youdao = "youdao"

def get_bot():
    config = read_config(config_path)
    
    BotKind = TranslateBotKind
    try:
        kind = BotKind(config['kind'])

        if kind is BotKind.Youdao:
            from youdao_bot import YoudaoBot
            return YoudaoBot(config['appKey'], config['appSecret'])

    except ValueError:
        raise ModuleNotFoundError(f"bot {kind} not found")
    except Exception as e:
        raise e