import numpy as np
import matplotlib.pyplot as plt
filename='fix_smalltest_output_result_1604067444'
shift=20
leng=9000
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

with open(f'{filename}.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [(float(x.strip())*4+1)/5 for x in content]

mean=[np.mean(content[i:i+shift]) for i in range(leng)]

print(mean)

plt.plot([i for i in range(leng)],mean)
plt.plot([i for i in range(leng)],[0.8372]*leng)

plt.xlabel('迭代次数',size=15)
plt.ylabel('系统服务可靠度',size=15)
plt.show()
# plt.savefig(f'plot_{filename}.jpg')