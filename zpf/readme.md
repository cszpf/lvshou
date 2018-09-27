# 程序文件说明
标签: 程序说明
---

- ./setting中文件说明
    1. label_tokens - python dict, 表示每一类中的tokens的Counter
    2. label_list - python list, 表示每一类别所包含的文档数
    3. id_to_label - python dict, 表示将id和类别的对应关系
    4. Vocab - python list, 表示词汇表
    5. W_L_DF -  pandas.DataFrame, 表示词-类别矩阵
    6. Vocab_bdc - python dict, 每一项表示词以及词对应的bdc值
    7. 保存{Label}.pk->该文件含有以上5个object

- 注意事项
    1. 由于违规类型之间存在交叉的现象，故应采用多个2分类的方法；因此需要计算多个bdc值

- 附录
    $$ bdc(t) = 1 - \frac{BH(t)}{log(|C|)}
    BH(t) = -\sum_{i=1}^{|C|}(P_{t,c_i}*log(P_{t,c_i}))
    p_{t,c_i} = \frac{p(t|c_i)}{\sum_{i=1}^{|C|}p(t|c_i)}
    p(t|c_i) = \frac{f(t,c_i)}{f(c_i)} $$
    $ f(t,c_i)表示词t在类别c_i中出现的次数;f(c_i)表示类别c_i所拥有的文档数 $