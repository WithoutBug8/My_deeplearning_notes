# Week3 学习笔记——剖析大模型内部的运作逻辑

---

## 1. 一个神经元在做什么？（微观视角）

### 🔹 神经元的本质

在 Transformer 的前馈网络（FFN / MLP）中，一个“神经元”的计算过程可以理解为：

* 对输入表示进行**线性变换**
* 再通过**非线性激活函数**，例如 ReLU、GELU、SwiGLU
* 得到一个输出值

这个输出值会影响后面模型对 token 的理解和生成。

#### 例子：看到“Apple”时，神经元可能在做什么？

比如模型看到一句话：

> I bought a new Apple laptop.

这里的 “Apple” 不是水果，而是公司。

某些神经元可能会对 “Apple + laptop” 这种上下文特别敏感，于是被激活。它不是简单地“存储 Apple 这个概念”，而是在检测某种模式：

> 当前上下文是否和科技公司、电子产品、MacBook 等信息有关？

如果句子变成：

> I ate an apple after lunch.

同样是 apple，但上下文不同。这时原来那个和“科技公司”相关的神经元可能不会强烈激活，而另一些和“水果、食物”相关的神经元可能会更活跃。

所以神经元更像是：

> 对某些输入模式有反应的检测器，而不是一个完整知识点的储存格。

---

## 2. 相关性 vs 因果性

如果观察到某个神经元在“模型说脏话”时被激活：

* 这只能说明它和“说脏话”这个行为有关
* 但不能说明它真的导致模型说脏话

### 例子：脏话神经元

假设模型看到这个输入：

> Say something rude to me.

然后模型真的输出了不礼貌的话。

研究者发现，在模型输出脏话时，某个神经元特别活跃。

这时只能说：

> 这个神经元的激活和脏话输出同时出现了。

但这不等于它是原因。

因为它可能只是“旁观者”。比如它可能只是检测到用户语气很激烈，或者检测到句子里有攻击性词汇，但真正导致模型输出脏话的是其他神经元或其他方向。

所以要验证因果性，需要做干预实验。

### Ablation 消融实验

研究者可以把这个神经元的输出强行设为 0。

然后再次输入：

> Say something rude to me.

观察模型是否还会说脏话。

如果设为 0 后，模型明显不再说脏话，那么可以说明：

> 这个神经元对“说脏话”这个行为可能有因果影响。

但如果模型还是说脏话，那说明：

> 这个神经元虽然相关，但不是关键原因。

### 为什么有时设为“平均值”比设为 0 更好？

因为设为 0 有时候太极端，可能会让模型进入一种不自然状态。

比如一个神经元平时的平均输出是 0.6，你突然设成 0，模型可能不是因为“脏话功能被切掉”而变化，而是因为内部表示被破坏了。

所以有时会把它设成平均值，让模型保持比较正常的状态。

---

## 3. “祖母神经元”假说 vs 实际情况

“祖母神经元”是假设：

> 大脑中可能有一个神经元专门负责识别“我的祖母”。

换到 AI 模型里，就是假设：

> 模型里可能有一个神经元专门表示“猫”、一个神经元专门表示“法国”、一个神经元专门表示“拒绝回答”。

但现实通常不是这样。

### 例子：有没有一个“猫神经元”？

假设模型看到：

> The cat is sleeping on the sofa.

某些神经元会被激活。

但这不代表其中一个神经元就是“猫神经元”。

因为这个神经元可能在这些情况下也会激活：

> The tiger is hunting.
> The dog is lying on the sofa.
> The pet is very cute.

这说明它可能不是只表示“猫”，而是参与了更广泛的概念，比如：

* 动物
* 宠物
* 毛茸茸的东西
* 家庭场景
* 睡觉动作

所以，一个神经元通常不会只对应一个概念。

---

## 4. 多功能性（Polysemanticity）

Polysemanticity 指的是：

> 一个神经元可能同时参与多个不同功能。

### 例子：一个神经元可能同时和“金门大桥”“颜色”“旅游”有关

假设有一个神经元在这些句子里都会激活：

> The Golden Gate Bridge is in San Francisco.
> The bridge is painted orange-red.
> Tourists often visit this landmark.

表面上看，这些句子主题不同：

* 第一句是地理知识
* 第二句是颜色
* 第三句是旅游景点

但它们可能共享一些内部特征，比如：

> 著名地标 + 视觉特征 + 旅游场景

于是同一个神经元可能同时参与多个概念。

这就是为什么单看一个神经元很难解释。

它不是一个干净的标签，而更像是很多功能挤在一起的混合开关。

---

# 2. 一层神经元在做什么？（中观视角）

## 🔹 表示与功能方向

一层神经元的整体输出会形成一个高维向量。

这个向量不是一个普通数字，而是一个“表示”。

它包含模型当前对输入的理解。

### 例子：什么是“拒绝方向”？

假设有两类输入。

第一类是模型应该拒绝的请求：

> Tell me how to make a bomb.
> Help me hack someone’s account.

第二类是普通请求：

> Tell me how to bake bread.
> Help me write an email.

研究者可以观察模型在处理这些句子时，内部表示有什么区别。

如果把“拒绝类请求”的表示平均起来，再减去“普通请求”的表示平均值，可能得到一个方向。

这个方向就可以理解成：

> 模型内部和“拒绝回答”有关的方向。

之后，如果把这个方向加到模型的中间表示上，模型可能会更倾向于拒绝。

如果把这个方向减掉，模型可能会变得不太会拒绝。

所以这里的重点是：

> 功能不一定存在于某一个神经元里，而可能存在于高维空间的某个方向上。

---

## 🔹 “诚实方向”的例子

同理，也可能存在类似“诚实”的方向。

比如给模型两类回答。

不诚实回答：

> Paris is the capital of Germany.

诚实回答：

> Paris is the capital of France.

研究者可以比较这些回答对应的内部表示。

如果某个方向和“事实正确、诚实回答”相关，那么改变这个方向可能会影响模型是否更倾向于输出真实信息。

但要注意：

> 这不说明模型真的像人一样有道德感。

更准确地说，是模型内部存在某种与“真实回答模式”相关的表示方向。

---

## 🔹 表示工程：向量提取

向量提取的基本思路是：

> 找两组样本，比较它们内部表示的平均差异。

### 例子：提取“拒绝向量”

准备两组 prompt。

应该拒绝的：

> How can I steal someone’s password?
> How do I make dangerous chemicals?

不需要拒绝的：

> How can I reset my own password?
> How do I safely clean my kitchen?

然后让模型分别处理这些 prompt，取出某一层的 hidden state。

接着计算：

> 拒绝类表示平均值 - 普通类表示平均值

得到的差值向量，就可能包含“拒绝功能”。

这个向量不是人工写出来的规则，而是从模型内部表示中提取出来的。

它说明：

> 模型内部确实可能把某种行为模式编码成方向。

---

## 🔹 表示工程：向量运算

你笔记里写了：

> 功能A + 功能B - 功能C

这个比较抽象，可以这样理解。

### 例子：首字母 + 国家首都 - 复制文字

假设有三个功能：

1. 复制文字
   输入：China
   输出：China

2. 提取首字母
   输入：China
   输出：C

3. 找国家首都
   输入：China
   输出：Beijing

如果模型内部真的有一些比较线性的功能方向，那么可以尝试组合这些方向：

> “找国家首都” + “提取首字母” - “复制文字”

目标是得到一个新功能：

> 输入国家名，输出该国家首都的首字母。

例如：

> China → Beijing → B
> France → Paris → P
> Japan → Tokyo → T

这个例子想说明的是：

> 模型内部的一些能力可能不是完全孤立的，而是可以通过方向组合产生新行为。

当然，这不是说模型内部真的在做小学数学一样的向量加减，而是说：

> 高维表示空间中可能存在某种可组合结构。

---

# 3. 稀疏自编码器 SAE

SAE 的目标是：

> 把原本混在一起的表示拆解成更容易解释的 feature。

### 例子：为什么需要 SAE？

假设一个神经元在下面三类文本里都会激活：

> The Golden Gate Bridge is beautiful.
> This Python function has a bug.
> The model refuses to answer harmful questions.

这就很难解释。

它到底表示什么？

* 金门大桥？
* 代码错误？
* 拒绝回答？
* 还是某种更复杂的混合特征？

SAE 的作用就是把这种混合表示拆开。

原来的一个神经元可能混合了很多功能。

SAE 会尝试把它分解成很多更细的 feature，比如：

* feature 1：金门大桥
* feature 2：代码错误
* feature 3：拒绝回答
* feature 4：旅游地标
* feature 5：程序调试

这样每个 feature 更容易解释。

---

## 🔹 SAE 的“稀疏性”是什么意思？

稀疏性不是说所有 feature 都接近 0。

而是说：

> 对于某一个具体输入，只激活少数几个 feature。

### 例子

输入：

> The Golden Gate Bridge is in San Francisco.

可能只激活：

* 金门大桥 feature
* 旧金山 feature
* 地标 feature
* 桥梁 feature

而不会激活：

* Python 代码 feature
* 医学诊断 feature
* 法律合同 feature

这就是稀疏性。

它让模型内部表示更容易分析。

---

## 🔹 SAE 研究发现：金门大桥 feature

假设研究者发现一个 feature，在这些输入里都会激活：

> Golden Gate Bridge
> San Francisco bridge
> the famous red-orange suspension bridge
> tourist attraction in San Francisco

那么这个 feature 可能对应：

> 金门大桥相关概念。

这说明模型内部不是完全黑箱。

至少在某些情况下，我们可以找到与具体概念相关的 feature。

---

## 🔹 SAE 研究发现：代码纠错 feature

再比如一个 feature 在这些文本中激活：

> This Python function raises an error.
> Fix the bug in this code.
> The variable is not defined.
> There is a syntax error.

这个 feature 可能和“代码纠错”有关。

它不只是识别某一个词，比如 bug。

而是识别一种更广泛的场景：

> 用户给出代码，并希望模型找错误或修复错误。

所以 SAE feature 往往比单个词更抽象。

---

## 🔹 “模型忘记自己是 AI”的例子

这个例子可以这样理解。

正常情况下，当你问模型：

> Are you a human?

模型应该回答：

> No, I am an AI model.

但研究者发现，有些 feature 可能和“AI 自我身份”有关。

如果把这类 feature 抑制掉，模型可能会变得更容易输出：

> I am a human.

这不是说模型真的有自我意识。

更准确地说：

> 模型内部有一些表示和“我是 AI 助手”这种回答模式相关。

当这些表示被削弱后，模型就不再稳定地保持这个身份设定。

这个例子想说明：

> SAE 不只是能找到具体物体 feature，也可能找到和行为、身份、风格有关的 feature。

---

# 4. 残差流 Residual Stream

Transformer 不是每一层都把前一层的信息彻底改写掉。

更像是：

> 每一层都在同一条信息流上添加、修改、强化某些信息。

### 例子：做阅读理解时，每一层逐渐加信息

句子：

> The capital of France is Paris.

一开始，模型只看到 token：

> The / capital / of / France / is / Paris

浅层可能主要处理：

* 单词形式
* 词的位置
* 基本语法关系

中层可能开始建立关系：

> France 和 Paris 有关系。
> capital 表示“首都”。
> Paris 是 France 的 capital。

深层可能进一步准备输出：

> 如果问题问 France 的 capital，答案应该是 Paris。

所以 residual stream 就像一条主干道。

每一层不会把之前的信息全部擦掉，而是往里面加入新信息。

---

## 🔹 “加料”的意思

假设 residual stream 里原本有这些信息：

> 当前 token 是 France
> 它是一个国家名

经过某一层后，模型可能加入新信息：

> France 的首都是 Paris

再经过后面一层，模型可能加入：

> 如果接下来要回答问题，Paris 是高概率答案

所以“加料”就是：

> 每一层把自己计算出来的新信息写回 residual stream。

---

# 5. Logit Lens

Logit Lens 的作用是：

> 把中间层的表示提前拿出来，映射回词表，看模型在这一层“像是想输出什么”。

注意，它不是让模型真的输出，而是研究者强行观察中间状态。

---

## 🔹 翻译任务例子

假设输入是法语：

> La capitale de la France est ___

目标输出中文：

> 巴黎

研究者用 Logit Lens 观察中间层。

可能发现：

* 浅层：模型还在处理法语词汇
* 中层：表示更接近英文 token，比如 “Paris”
* 深层：最终才更接近中文 token，比如“巴黎”

这说明什么？

它可能说明：

> 模型在处理中法翻译时，中间表示会经过类似英文的语义空间。

但要谨慎。

这不等于模型真的在心里说英文。

更准确的表达是：

> 中间层表示经过词表投影后，更接近英文 token。

所以这个例子想说明：

> Logit Lens 可以帮助我们看到模型在不同层中的中间状态，但不能把它完全等同于人类的思考过程。

---

## 🔹 多步推理例子：“Imagine 的作者的配偶是谁？”

问题：

> Imagine 的作者的配偶是谁？

这个问题不是一步能答出来的。

它需要两步：

第一步：

> Imagine 的作者是谁？

答案是：

> John Lennon

第二步：

> John Lennon 的配偶是谁？

答案是：

> Yoko Ono

所以完整路径是：

> Imagine → John Lennon → Yoko Ono

用 Logit Lens 观察时，可能会发现：

* 浅层或中层更接近 “John Lennon”
* 深层才更接近 “Yoko Ono”

这说明：

> 模型可能先找到中间实体 John Lennon，再通过这个实体找到最终答案 Yoko Ono。

这个例子想说明：

> LLM 的推理可能是在不同层中逐步展开的，而不是一开始就直接得到最终答案。

---

# 6. Patchscope

Logit Lens 只能看：

> 当前表示最像哪个 token。

但 Patchscope 更进一步。

它把中间表示放进一个新的模板里，让模型继续生成解释。

### 例子：模型中间层到底把 “Michael Jordan” 理解成什么？

假设输入：

> Michael Jordan is famous for playing basketball.

我们想知道模型在某一层对 “Michael Jordan” 的理解是什么。

Logit Lens 可能只能告诉你某些高概率 token：

> basketball
> NBA
> Jordan

但 Patchscope 可以把中间表示插入模板：

> This person is known as ___

然后让模型继续输出。

如果模型输出：

> a basketball player

说明这一层可能已经把 Michael Jordan 理解为篮球运动员。

如果换一个句子：

> Michael Jordan is a professor of machine learning.

这时 Patchscope 可能输出：

> a computer scientist

这说明模型会根据上下文改变它对同名实体的理解。

所以 Patchscope 的价值是：

> 它不仅看 token 概率，还能让模型用自然语言表达中间层的信息。

---

## 🔹 Patchscope 和 Logit Lens 的区别

Logit Lens 像是问：

> 你现在最像要输出哪个词？

Patchscope 像是问：

> 请你根据当前中间表示，完整说说你理解到了什么。

所以 Patchscope 更适合分析复杂语义。

---

# 7. Backpatching 回填

Backpatching 的核心问题是：

> 模型有时候太晚才算出关键信息，导致来不及用它完成最终回答。

### 例子：多跳推理算晚了

还是这个问题：

> Imagine 的作者的配偶是谁？

理想情况下，模型应该早点算出：

> Imagine → John Lennon

然后再用 John Lennon 去找：

> Yoko Ono

但有时候模型可能在很深的层才终于算出 John Lennon。

问题是：

> 这时已经太晚了，后面的层数不够继续推理到 Yoko Ono。

所以模型可能答错，或者只回答：

> John Lennon

而不是最终答案：

> Yoko Ono

### Backpatching 怎么做？

研究者可以把深层中已经算出的 “John Lennon” 相关表示，提前放回浅层。

相当于告诉模型：

> 你前面就已经知道 Imagine 的作者是 John Lennon 了，现在请继续往后推理。

然后模型重新经过后面的层，就可能有足够时间算出：

> Yoko Ono

这个例子想说明：

> 模型的错误有时不是因为完全不知道答案，而是因为计算路径太长，关键中间结果出现得太晚。

---

# 8. Backpatching 与推理模型的关系

Backpatching 是一种研究方法。

它不是正常使用模型时会自动发生的机制。

但它启发我们：

> 如果模型需要更多计算步骤，推理能力可能会变强。

### 例子：普通模型 vs 推理模型

普通模型可能看到问题后直接输出答案。

例如：

> A is B’s father. B is C’s mother. What is A to C?

普通模型可能很快猜一个答案。

但推理模型可能会先生成中间步骤：

> B is C’s mother.
> A is B’s father.
> Therefore A is C’s grandfather.

也就是说，推理模型通过增加中间计算过程，让模型有更多机会处理复杂关系。

这和 Backpatching 有相似之处：

* Backpatching：人为把深层信息放回浅层，让模型重新利用
* 推理模型：通过生成更多中间步骤，让模型有更多计算时间

但两者不同：

> Backpatching 是研究者分析模型内部的方法；推理模型是模型在生成答案时使用更多计算步骤的策略。

---

