import numpy as np
import matplotlib.pyplot as plt
filename='flex_smalltest_output_result_move_amount_limit3_1604317499.txt'
shift=20
leng=20000
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [float(x.strip())/4 for x in content]

mean=[np.mean(content[i:i+shift]) for i in range(leng)]

print(mean)

plt.plot([i for i in range(leng)],mean)
plt.plot([i for i in range(leng)],[0.804]*leng)

plt.xlabel('迭代次数',size=15)
plt.ylabel('系统服务可靠度',size=15)
# plt.show()
plt.savefig('plot_flex_smalltest_output_result_move_amount_limit3_1604317499.jpg')