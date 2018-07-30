import cProfile
import re
import pstats
from MTSVRC import *

data_path = "/media/guo/搬砖BOY/dataset/"  # "D:/dataset/"#/media/guo/搬砖BOY/dataset/"
label_train = "/media/guo/搬砖BOY/English/trainEnglish.txt"  # ("D:/dataset/English/trainEnglish.txt")
label_val = "/media/guo/搬砖BOY/English/valEnglish.txt"  # ("D:/dataset/English/valEnglish.txt")
mtsvrc = MTSVRC(data_path, label_train, label_val)
print("cuda:" + str(torch.cuda.is_available()))
# mtsvrc.data_check(batch_size=8,filein_batch=64,learning_rate=0.05)

cProfile.run("mtsvrc.data_check(batch_size=1,filein_batch=16,learning_rate=0.05)",sort="cumulative")
# p = pstats.Stats("result.out")
#
# # strip_dirs(): 去掉无关的路径信息
# # sort_stats(): 排序，支持的方式和上述的一致
# # print_stats(): 打印分析结果，可以指定打印前几行
#
# # 和直接运行cProfile.run("test()")的结果是一样的
# p.strip_dirs().sort_stats(-1).print_stats()
