# 2017年12月17日讨论记录

* 开始时间：17:00
* 时长：1.5h
* 参加人员：朱宇森，李倩怡，郭潇潇，姜家隆，申云逸

## 讨论内容

* 确定讨论形式：线上+线下，GitHub存档，合适时间在BBS发布。
* 基本思路：根据基因组中未知功能部分的进化优势和劣势建立定量模型进行模拟进化实验，观察最终获胜群体的组成与性质。
* 重要假设：这些未知功能部分中的很大部分在当前环境下无生物功能，属于演化遗迹或为提高适应性付出的代价。

### 模型细节

* 基因组用二进制序列表示
* 长基因组的劣势是降低复制速率，以及有可能在存活率上有惩罚。
* 定义特定基因序列是“好的”，能够降低死亡率/增加后代数目
    * 卷积实现打分
 
