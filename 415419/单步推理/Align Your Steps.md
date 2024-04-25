
Align Your Steps:    
Optimizing Sampling Schedules in Diffusion Models





# 论文信息
英伟达多伦多ai实验室     
1 NVIDIA2 University of Toronto3 Vector Institute    

[Submitted on 22 Apr 2024]     
Align Your Steps: Optimizing Sampling Schedules in Diffusion Models


DM 的一个关键缺点是采样速度慢，依赖于通过大型神经网络进行的许多顺序函数评估。从 DM 采样可以被视为通过一组离散化的噪声水平（称为采样计划）求解微分方程。虽然过去的工作主要集中在推导有效的求解器，但很少关注寻找最佳采样计划，并且整个文献都依赖于手工设计的启发式方法。在这项工作中，我们首次提出“调整您的步骤”，这是一种优化 DM 采样计划以获得高质量输出的通用且有原则的方法。我们利用随机微积分的方法，找到针对不同求解器、经过训练的 DM 和数据集的最佳调度。我们使用各种不同的求解器在多个图像、视频以及 2D 玩具数据合成基准上评估我们的新颖方法，并观察到我们的优化计划在几乎所有实验中都优于以前的手工计划。我们的方法展示了采样计划优化的未开发潜力，特别是在少步合成机制中。

Below, we showcase some text-to-image examples that illustrate how using an optimized schedule can generate images with more visual details and better text-alignment given the same number of forward evaluations (NFEs). We provide side-by-side comparisons between our optimized schedules against two of the most popular schedules used in practice (EDM and Time-Uniform). All images are generated with a stochastic (casino) or deterministic (lock) version of DPM-Solver++(2M) with 10 steps. Hover over the images for zoom-ins.

![alt text](<assets/Align Your Steps/截屏2024-04-25 11.39.12.png>)









Stable Video Diffusion     
We also studied the effect of optimized schedules in video generation using the open-source image-to-video model Stable Video Diffusion. We find that using optimized schedules leads to more stable videos with less color distortions as the video progresses. Below we show side-by-side comparisons of videos generated with 10 DDIM steps using the two different schedules.





# 实践
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html     
好像和 pixart dmd 一样，都会在timestep进行下手手动设置    
确实类似，但是pixart dmd我只测过单步，然后取400.    
英伟达这里好像倾向于10步     
可以取更广范围     












# 结尾