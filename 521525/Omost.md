Omost是一个将LLM的编码能力转换为图像生成（或更准确地说，图像合成）能力的项目。

这个名字（发音：几乎）有两个含义：1）每次使用Omost后，你的形象几乎就在那里;2）平均“omni”（多模态），意味着我们希望充分利用它。OmostOmost

Omost提供了LLM模型，这些模型将编写代码，以使用Omost的虚拟代理来组合图像视觉内容。这可以通过图像生成器的特定实现来呈现，以实际生成图像。CanvasCanvas

目前，我们提供了 3 个基于 Llama3 和 Phi3 变体的预训练 LLM 模型（另请参阅本页末尾的模型说明）。

所有模型都使用混合数据进行训练：（1） 包括 Open-Images 在内的多个数据集的地面实况注释，（2） 通过自动注释图像提取数据，（3） 来自 DPO（直接偏好优化，“代码是否可以由 python 3.10 编译”作为直接偏好）的强化，以及 （4） 来自 OpenAI GPT4o 多模态功能的少量调整数据。

# 输入1
用户     
生成勇士与龙的激烈战斗的图像


    # Initialize the canvas
    canvas = Canvas()

    # Set a global description for the canvas
    canvas.set_global_description(
        description='A fierce battle between warriors and a dragon.',
        detailed_descriptions=[
            'In this intense scene, a group of fierce warriors is engaged in an epic battle with a mighty dragon.',
            'The warriors, clad in armor and wielding swords and shields, are positioned on the left side of the image.',
            'Their expressions are determined and focused, reflecting their resolve to defeat the dragon.',
            'The dragon, with its massive wings spread wide and its fiery breath illuminating the scene, dominates the center of the image.',
            'Its scales glisten with a metallic sheen, and its eyes burn with a fierce intensity.',
            'The background is a dark, stormy sky with flashes of lightning, adding to the drama and tension of the battle.',
            'The ground is littered with debris and the remnants of previous battles, enhancing the sense of a long and brutal conflict.',
            'The overall atmosphere is one of chaos and intensity, with the warriors and the dragon locked in a fierce struggle for survival.',
        ],
        tags='battle, warriors, dragon, fierce, armor, swords, shields, determined, focused, epic, intense, metallic, glistening, fiery breath, stormy sky, lightning, debris, conflict, struggle, survival, chaos, tension, drama, wings, scales, eyes, burning, illuminated',
        HTML_web_color_name='darkslategray',
    )

    # Add a group of fierce warriors.
    canvas.add_local_description(
        location='on the left',
        offset='no offset',
        area='a large horizontal area',
        distance_to_viewer=5.0,
        description='A group of fierce warriors.',
        detailed_descriptions=[
            'The warriors, clad in gleaming armor, are positioned on the left side of the image.',
            'They are armed with swords, shields, and spears, ready for battle.',
            'Their faces are set with determination and focus, reflecting their resolve to defeat the dragon.',
            'Some warriors are in mid-action, swinging their swords or shields, while others stand ready to strike.',
            'The armor they wear is intricately designed, with symbols and patterns that signify their rank and bravery.',
            'The ground beneath them is uneven and littered with debris, adding to the sense of a chaotic and intense battle.',
        ],
        tags='warriors, armor, swords, shields, spears, determined, focused, mid-action, intricate design, symbols, patterns, rank, bravery, uneven ground, debris, chaotic, intense, battle',
        atmosphere='Determined and focused, ready for the fierce battle.',
        style='Highly detailed and dynamic, capturing the intensity of the warriors.',
        quality_meta='High resolution with intricate details and dynamic poses.',
        HTML_web_color_name='darkgoldenrod',
    )

    # Add a mighty dragon.
    canvas.add_local_description(
        location='in the center',
        offset='no offset',
        area='a large square area',
        distance_to_viewer=7.0,
        description='A mighty dragon.',
        detailed_descriptions=[
            'The dragon is a massive creature, dominating the center of the image with its wide-spread wings and fiery breath.',
            'Its scales glisten with a metallic sheen, reflecting the light from its fiery breath.',
            "The dragon's eyes burn with a fierce intensity, and its teeth are sharp and menacing.",
            'The wings of the dragon are powerful and spread wide, casting shadows over the battlefield.',
            'The dragon’s body is muscular and covered in protective scales, with a long, sinewy tail that adds to its formidable appearance.',
            'The fiery breath of the dragon illuminates the scene, casting a reddish glow over the warriors and the battlefield.',
        ],
        tags='dragon, massive, wings, fiery breath, glistening scales, metallic sheen, fierce eyes, sharp teeth, powerful wings, shadows, battlefield, muscular body, protective scales, sinewy tail, formidable, reddish glow, illumination',
        atmosphere='Intense and menacing, with a powerful presence.',
        style='Epic and dramatic, emphasizing the grandeur and danger of the dragon.',
        quality_meta='High resolution with dramatic lighting and detailed textures.',
        HTML_web_color_name='firebrick',
    )

    # Add a stormy sky with flashes of lightning.
    canvas.add_local_description(
        location='on the top',
        offset='no offset',
        area='a large horizontal area',
        distance_to_viewer=10.0,
        description='A stormy sky with flashes of lightning.',
        detailed_descriptions=[
            'The background of the image is a dark, stormy sky filled with swirling clouds and flashes of lightning.',
            'The sky is turbulent, with clouds dark and foreboding, adding to the dramatic tension of the battle.',
            'The lightning flashes illuminate the scene, casting sharp, brief lights over the warriors and the dragon.',
            'The stormy sky creates a sense of chaos and unpredictability, heightening the intensity of the battle below.',
            'The overall atmosphere is one of impending doom and relentless conflict, with the storm mirroring the fierce struggle between the warriors and the dragon.',
        ],
        tags='stormy sky, dark clouds, lightning, turbulent, foreboding, dramatic tension, illumination, chaos, unpredictability, intensity, impending doom, relentless conflict, battle, warriors, dragon, swirling clouds, sharp lights, brief lights',
        atmosphere='Chaotic and intense, mirroring the fierce battle below.',
        style='Dramatic and turbulent, emphasizing the conflict and tension.',
        quality_meta='High resolution with dynamic lighting and detailed cloud textures.',
        HTML_web_color_name='midnightblue',
    )

    # Add a debris-covered battlefield.
    canvas.add_local_description(
        location='on the bottom',
        offset='no offset',
        area='a large horizontal area',
        distance_to_viewer=5.0,
        description='A debris-covered battlefield.',
        detailed_descriptions=[
            'The ground of the battlefield is littered with debris, remnants of previous battles.',
            'Broken weapons, shattered shields, and scattered armor pieces are strewn across the battlefield.',
            'The terrain is uneven, with patches of mud and dirt, adding to the sense of a long and brutal conflict.',
            'The debris-covered battlefield enhances the chaotic and intense atmosphere of the scene, reflecting the ferocity and duration of the battle.',
            'The overall appearance is one of destruction and turmoil, with the remnants of previous battles serving as a grim reminder of the ongoing struggle.',
        ],
        tags='battlefield, debris, broken weapons, shattered shields, scattered armor, uneven terrain, mud, dirt, brutal conflict, chaos, intensity, destruction, turmoil, previous battles, ongoing struggle, remnants, ferocity, duration',
        atmosphere='Chaotic and intense, reflecting the ferocity of the battle.',
        style='Detailed and realistic, emphasizing the destruction and chaos.',
        quality_meta='High resolution with detailed textures and realistic debris.',
        HTML_web_color_name='darkolivegreen',
    )


## 输入2
用户    
把龙变成恐龙



# 相关工作
另请阅读...

DOCCI：连接和对比图像的描述

（RPG-扩散大师）掌握文本到图像的传播：使用多模态 LLM 重新字幕、规划和生成

MultiDiffusion：融合扩散路径以生成受控图像

sd-webui-regional-prompter





# 结尾