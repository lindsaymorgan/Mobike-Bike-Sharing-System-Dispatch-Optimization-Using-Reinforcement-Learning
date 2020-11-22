import numpy as np
import matplotlib.pyplot as plt
filename='subgraph32_real_output_result_carnum_2_carbatch5_1604414772.txt'
shift=10
leng=20000
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

with open(f'{filename}.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [float(x.strip())/4 for x in content]

mean=[np.mean(content[i:i+shift]) for i in range(leng)]

print(mean)

plt.plot([i for i in range(leng)],mean)
# plt.plot([i for i in range(leng)],[0.7964]*leng)

plt.xlabel('迭代次数',size=15)
plt.ylabel('系统服务可靠度',size=15)
plt.savefig(f'plot_{filename}.jpg')