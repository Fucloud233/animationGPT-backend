import sys; sys.path.append("./server")

from utils.translate import get_bot

if __name__ == '__main__':

    bot = get_bot()
    
    # bot.translate("你好")
    flag = bot.check()
    if flag:
        print(bot.translate("你好"))
    else:
        print(flag)
