## Brainstorm discussion Dec.23



1. **基因序列和功能的对应**：目前简单模型只考虑一个功能

2. **求和/指数**：指数再求和，先考虑环境选择的序列长度相同。（但是指数再求和很快就都灭绝了？）*属于代码问题，已经解决，还可以尝试不同的cost-function形式做系统分析*

3. **初步模拟结果**：

   - 目前的模拟中不同个体产生后代的概率是一样的以防止灭绝，最后存活的个体数是基本一定的，初步模拟结果中随着代数的增加基因组的长度是可能收敛的。


   - 长度均值和代数之间的关系，有平台期，但持续增加。随代数增加，基因组长度有跳跃，看起来是插转座子引起。

4. **junk gene**：最后基因组中剩下的序列中哪些gene是历史环境需求的，可以记录每代的分数。

5. **基因组长度分两拨**：猜想环境变化率高，转座子移动概率低会出现这种结果。短基因组获得优势依赖于具体的cost-function形式。

