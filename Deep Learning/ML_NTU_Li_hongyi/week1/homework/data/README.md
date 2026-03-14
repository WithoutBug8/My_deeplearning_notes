# 关于数据集的说明

### `covid.train`这个数据集的各列说明

这个数据集中有各个州 + 5天/周的疫情数据。 
- 你要做的预测下一天/周的疫情状况

以下是各个列的具体含义

| 变量                   | 含义                               |
| -------------------- | -------------------------------- |
| cli                  | COVID-like illness（类似新冠症状比例）     |
| ili                  | influenza-like illness（类似流感症状比例） |
| hh_cmnty_cli         | 家庭或社区出现类似症状                      |
| nohh_cmnty_cli       | 家庭或社区没有症状                        |
| wearing_mask         | 戴口罩比例                            |
| travel_outside_state | 跨州旅行比例                           |
| work_outside_home    | 外出工作比例                           |
| shop                 | 外出购物比例                           |
| restaurant           | 去餐厅比例                            |
| spent_time           | 与他人接触时间                          |
| large_event          | 参加大型活动比例                         |
| public_transit       | 乘坐公共交通比例                         |
| anxious              | 焦虑比例                             |
| depressed            | 抑郁比例                             |
| worried_finances     | 担心经济问题                           |
| tested_positive      | 新冠检测阳性比例                         |

### 这周作业的收获
1. 学会了怎么调整神经网络的参数
2. 对神经网络的训练过程理解更加清晰了

### 仍然存在的问题

- 对于调参获得好的结果仍然有疑问，如何控制各个参数对我来说仍然是一个困难的问题