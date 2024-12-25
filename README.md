这是2024年YUNA运维部寒假作业（其一）
本次作业选题是HPCG的部署和优化，于WSL上部署完成的时间为2024年12月19日。
部署参考了以下文章：
https://www.cnblogs.com/lijiaji/p/14283958.html
运行结果的位置是：hpcg/setup/build/bin/2024.12.19
注：hpcg运行时，设备性能上限设置为：四个CPU核心，4GB内存，4GB交换分区大小
下面是程序简介，参考了chatgpt的回答，最后会附上聊天分享链接：
程序功能介绍：这个程序实现了HPCG基准测试的主要功能，通过构建几何问题模型、生成线性系统、设置问题并通过共轭梯度算法求解稀疏线性方程组后进行验证以评估性能，其采用MPI进行并行计算。
核心代码介绍：
ComputeSPMV函数是用于计算y=Ax，其中A为稀疏矩阵，x，y为向量，这个函数是CG算法的核心，并且计算密集，占到了总运行时间的40%-50%，所以test1主要优化方向是该函数。
ComputeDotProduct函数是用于计算两个向量x，y的内积，这个函数需要全局归约操作且涉及MPI进程间的通信，故test2主要优化方向是该函数。
ComputeMG函数功能是实现多重网格预处理，其包含多个网格层次且涉及复杂的网格间传输，故test3的主要优化方向是该函数。
ComputeWAXPBY函数是用于计算w=αx+βy的，其计算简单规则且易于向量化，故test4优化方向为该函数。
test5是将上述四个函数都进行优化后的版本。
注：所有优化方案及实行皆由cursor完成。
chatgpt聊天链接：https://chatgpt.com/share/6767c999-8e5c-8003-824c-cc36b83a8337