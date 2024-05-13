Align Your Steps:    
Optimizing Sampling Schedules in Diffusion Models


# 论文信息
英伟达多伦多ai实验室     
1 NVIDIA2 University of Toronto3 Vector Institute    

[Submitted on 22 Apr 2024]     
Align Your Steps: Optimizing Sampling Schedules in Diffusion Models    
https://arxiv.org/abs/2404.14507     



DM 的一个关键缺点是采样速度慢，依赖于通过大型神经网络进行的许多顺序函数评估。从 DM 采样可以被视为通过一组离散化的噪声水平（称为采样计划）求解微分方程。虽然过去的工作主要集中在推导有效的求解器，但很少关注寻找最佳采样计划，并且整个文献都依赖于手工设计的启发式方法。在这项工作中，我们首次提出“调整您的步骤”，这是一种优化 DM 采样计划以获得高质量输出的通用且有原则的方法。我们利用随机微积分的方法，找到针对不同求解器、经过训练的 DM 和数据集的最佳调度。我们使用各种不同的求解器在多个图像、视频以及 2D 玩具数据合成基准上评估我们的新颖方法，并观察到我们的优化计划在几乎所有实验中都优于以前的手工计划。我们的方法展示了采样计划优化的未开发潜力，特别是在少步合成机制中。

Below, we showcase some text-to-image examples that illustrate how using an optimized schedule can generate images with more visual details and better text-alignment given the same number of forward evaluations (NFEs). We provide side-by-side comparisons between our optimized schedules against two of the most popular schedules used in practice (EDM and Time-Uniform). All images are generated with a stochastic (casino) or deterministic (lock) version of DPM-Solver++(2M) with 10 steps. Hover over the images for zoom-ins.

![alt text](<assets/Align Your Steps/截屏2024-04-25 11.39.12.png>)





Stable Video Diffusion     
We also studied the effect of optimized schedules in video generation using the open-source image-to-video model Stable Video Diffusion. We find that using optimized schedules leads to more stable videos with less color distortions as the video progresses. Below we show side-by-side comparisons of videos generated with 10 DDIM steps using the two different schedules.



# 原理
![alt text](<assets/Align Your Steps/image.png>)    
扩散模型 (DM) 已证明自己是极其可靠的概率生成模型，可以生成高质量的数据。它们已成功应用于图像合成、图像超分辨率、图像到图像翻译、图像编辑、修复、视频合成、文本到 3D 生成，甚至规划planning等应用。     
然而，从 DM 中采样相当于逆时求解生成随机或常微分方程 (SDE/ODE)，并且需要通过大型神经网络进行多次顺序前向传递，从而限制了其实时适用性。   
sampling from DMs corresponds to solving a generative Stochastic or Ordinary Differential Equation (SDE/ODE) in reverse time and requires multiple sequential forward passes through a large neural network     

![alt text](<assets/Align Your Steps/image-1.png>)    
![alt text](<assets/Align Your Steps/image-2.png>)     

Assuming that 𝑃𝑡𝑟𝑢𝑒 represents the distribution of running the reverse-time SDE (defined by the learnt model) exactly, and 𝑃𝑑𝑖𝑠𝑐 represents the distribution of solving it with Stochastic-DDIM and a sampling schedule, using the Girsanov theorem an upper bound can be derived for the Kullback-Leibler divergence between these two distributions   
![alt text](<assets/Align Your Steps/image-4.png>)    
![alt text](<assets/Align Your Steps/image-5.png>)   

![alt text](<assets/Align Your Steps/image-6.png>)    
![alt text](<assets/Align Your Steps/image-7.png>)    

这些示例说明了在给定相同数量的前向评估 (NFE) 的情况下，如何使用优化的计划生成具有更多视觉细节和更好文本对齐的图像。我们将优化的计划与实践中使用的两种最流行的计划（EDM 和 Time-Uniform）进行并排比较。所有图像都是用随机或确定性 版本的 DPM-Solver++(2M)，有 10 个步骤。




原理：   
Optimizing Sampling Schedules in Diffusion Models    
基于karras优化 DM 采样计划以获得高质量输出的通用且有原则的方法     
即在scheduler上做改进。类似于most popular schedules used in practice (EDM and Time-Uniform).    

例如：   
英伟达在DPM-Solver++(2M) karras上算出AYS采样timesteps为    
timesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]    
其他和正常采样一样    










# 实践
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/     
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html     
好像和 pixart dmd 一样，都会在timestep进行下手手动设置    
确实类似，但是pixart dmd我只测过单步，然后取400.    
英伟达这里好像倾向于10步     
可以取更广范围     


10步推理采用AYS效果比Karras稍好，AYS偶尔效果不好。    
AYS和sgm-uniform效果相近。少步和多步的生成质量感觉还可以。    


# 其他知识
## 启发式算法
启发式算法一般用于解决NP-hard问题，其中NP是指非确定性多项式。   
启发式算法是相对于最优化算法提出的，是基于直观或者经验构造的算法，在可接受的开销（时间和空间）内给出待解决组合优化问题的一个可行解。    

启发法（heuristics，又译作：策略法、助发现法、启发力）    
启发法不能保证问题解决的成功，但这种方法比较省力。它有以下几种策略：1、手段－目的分析：就是将需要达到问题的目标状态分成若干子目标，通过实现一系列的子目标最终达到总的目标；2、逆向搜索：就是从问题的目标状态开始搜索直至找到通往初始状态的通路或方法；3、爬山法：采用一定的方法逐步降低初始状态和目标状态的距离，以达到问题解决的一种方法。    

![alt text](<assets/Align Your Steps/image-3.png>)    


## Girsanov theorem
在概率论中，吉尔萨诺夫定理说明了随机过程如何随着测度的变化而变化。该定理在金融数学理论中尤其重要，因为它告诉我们如何从物理度量（描述基础工具（例如股价或利率）采用特定值或多个值的概率）转换为风险中性衡量标准，是评估标的 衍生品价值的非常有用的工具。    
In probability theory, the Girsanov theorem tells how stochastic processes change under changes in `measure`. The theorem is especially important in the theory of financial mathematics as it tells how to convert from the physical measure, which describes the probability that an underlying instrument (such as a share price or interest rate) will take a particular value or values, to the risk-neutral measure which is a very useful tool for evaluating the value of derivatives on the underlying.     

这种类型的结果首先由 Cameron-Martin 在 20 世纪 40 年代被证明，并由Igor Girsanov在 1960 年被证明。随后它们被扩展到更一般的过程类别，最终形成了 Leenglart (1977) 的一般形式。    
吉尔萨诺夫定理在随机过程的一般理论中很重要，因为它得出了关键结果：如果Q是相对于P绝对连续的测度，则每个P半鞅都是Q半鞅。   
Girsanov's theorem is important in the general theory of stochastic processes since it enables the key result that if Q is a measure that is absolutely continuous with respect to P then every P-semimartingale is a Q-semimartingale.    

金融应用   
该定理可用于在 Black-Scholes 模型中显示唯一的风险中性度量，即衍生品的公允价值是贴现预期值 Q 的度量

Application to Langevin equations    
Another application of this theorem, also given in the original paper of Igor Girsanov, is for stochastic differential equations. Specifically, let us consider the equation     









# 结尾