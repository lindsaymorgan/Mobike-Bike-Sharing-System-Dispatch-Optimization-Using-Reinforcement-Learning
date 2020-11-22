import numpy as np
import matplotlib.pyplot as plt

filename='not_equal_start_real_output_result_carnum_5_carbatch10_movelimit10_1604542587'
shift=20
leng=10000
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

with open(f'{filename}.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [float(x.strip()) for x in content]

mean=[np.mean(content[i:i+shift]) for i in range(leng)]

print(mean)

plt.plot([i for i in range(leng)],mean)
# plt.plot([i for i in range(leng)],[0.7964]*leng)

plt.xlabel('迭代次数',size=15)
plt.ylabel('系统服务可靠度',size=15)
plt.savefig(f'plot_{filename}.jpg')