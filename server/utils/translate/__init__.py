import sys; sys.path.append("./server/utils/translate")
import json
from enum import Enum
import logging

config_path = "./configs/translate.json"

def read_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    
class TranslateBotKind(Enum):
    Youdao = "youdao"

def get_bot():
    try:
        config = read_config(config_path)
    except FileNotFoundError as e:
        logging.error(f"{config_path} not found")
        from null_bot import NullBot

        return NullBot()
    
    BotKind = TranslateBotKind
    try:
        kind = BotKind(config['kind'])
        config = config[kind.value]
        if kind is BotKind.Youdao:
            from youdao_bot import YoudaoBot

            return YoudaoBot(config['appKey'], config['appSecret'])

    except ValueError:
        raise ModuleNotFoundError(f"bot {kind} not found")
    except Exception as e:
        raise e