import numpy as np
import matplotlib.pyplot as plt
filename='fix_smalltest_output_result_1603248561.txt'
shift=10
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [float(x.strip()) for x in content]

mean=[np.mean(content[i:i+shift]) for i in range(5000)]

print(mean)

plt.plot([i for i in range(5000)],mean)
plt.plot([i for i in range(5000)],[0.796]*5000)

plt.xlabel('迭代次数',size=15)
plt.ylabel('系统服务可靠度',size=15)
plt.savefig('plot_fix_smalltest_output_result_1603248561.jpg')