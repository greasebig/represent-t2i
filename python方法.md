# 重写覆盖
你可以使用Python中的装饰器、上下文管理器或猴子补丁技术来在运行时动态替换类的方法，而无需修改原始代码。以下是几种常用的实现方法：

方法1：使用猴子补丁（Monkey Patching）
猴子补丁是一种动态替换类或模块中的方法的技术。以下是一个例子：

    # 假设这是原始类定义
    class MyClass:
        def original_method(self):
            print("This is the original method")

    # 定义一个新方法
    def new_method(self):
        print("This is the new method")

    # 替换方法
    MyClass.original_method = new_method

    # 测试
    obj = MyClass()
    obj.original_method()  # 输出: This is the new method
方法2：使用上下文管理器
上下文管理器可以用于在特定上下文中临时替换类的方法。可以使用contextlib模块中的contextmanager装饰器来创建一个上下文管理器。


    import contextlib

    class MyClass:
        def original_method(self):
            print("This is the original method")

    def new_method(self):
        print("This is the new method")

    @contextlib.contextmanager
    def replace_method(klass, method_name, new_method):
        original_method = getattr(klass, method_name)
        setattr(klass, method_name, new_method)
        try:
            yield
        finally:
            setattr(klass, method_name, original_method)

    # 测试
    obj = MyClass()
    obj.original_method()  # 输出: This is the original method

    with replace_method(MyClass, 'original_method', new_method):
        obj.original_method()  # 输出: This is the new method

    obj.original_method()  # 输出: This is the original method
方法3：使用装饰器
你可以创建一个装饰器来动态替换方法。


    from functools import wraps

    def replace_method_decorator(new_method):
        def decorator(method):
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                return new_method(self, *args, **kwargs)
            return wrapper
        return decorator

    class MyClass:
        @replace_method_decorator(lambda self: print("This is the new method"))
        def original_method(self):
            print("This is the original method")

    # 测试
    obj = MyClass()
    obj.original_method()  # 输出: This is the new method
在上面的例子中，装饰器会在类定义时替换掉原始方法。

方法4：动态修改实例方法
如果你只想在特定实例上替换方法，而不是整个类，可以直接在实例上绑定新方法。


    class MyClass:
        def original_method(self):
            print("This is the original method")

    def new_method(self):
        print("This is the new method")

    # 测试
    obj = MyClass()
    obj.original_method()  # 输出: This is the original method

    import types
    obj.original_method = types.MethodType(new_method, obj)
    obj.original_method()  # 输出: This is the new method

    # 新创建的实例仍然使用原始方法
    another_obj = MyClass()
    another_obj.original_method()  # 输出: This is the original method
选择哪种方法取决于你的具体需求，例如你是否需要全局替换方法，还是只在特定上下文中临时替换。




# ... 相对导入
from modules import devices, scripts：

绝对导入：这是一种绝对导入方式，表示从当前顶层的包开始导入modules模块，然后从modules模块中导入devices和scripts子模块。    
适用场景：当你确定modules在Python路径中，并且不会改变位置时使用。


from ..modules import devices, scripts：

相对导入（上一级目录）：这是一种相对导入方式，表示从当前模块的上一级目录开始寻找modules模块，然后从modules模块中导入devices和scripts子模块。   
适用场景：用于包内部模块之间的导入，当前模块和modules模块位于同一父目录下。

from ...modules import devices, scripts：

相对导入（上上一级目录）：这是一种相对导入方式，表示从当前模块的上上一级目录开始寻找modules模块，然后从modules模块中导入devices和scripts子模块。    
适用场景：用于包内部更深层次的模块之间的导入，当前模块和modules模块位于不同的层级下。

    project/
    ├── package/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── modules/
    │   │   ├── __init__.py
    │   │   ├── devices.py
    │   │   └── scripts.py
    │   └── subpackage/
    │       ├── __init__.py
    │       ├── submodule.py
            └── another_subpackage/
                └── submodule.py

在main.py中：

使用from modules import devices, scripts来导入modules中的子模块。

在subpackage/submodule.py中：

使用from ..modules import devices, scripts来导入上一级目录的modules中的子模块。

如果你在更深的层级中（比如subpackage/another_subpackage/submodule.py），     
则可以使用from ...modules import devices, scripts。


## 报错


ImportError: attempted relative import beyond top-level package 


错误原因

首先，让我们了解一下相对导入的概念。相对导入是指从当前模块的位置出发，根据相对关系进行导入。经常使用的相对导入语法包括使用单个点（.）表示当前目录以及使用双点（..）表示父目录。 然而，当我们的代码位于顶层包之外时，试图进行相对导入就会引发"attempted relative import beyond top-level package"错误。这是因为相对导入需要一个确定的顶层包来构建相对路径。如果我们的代码在顶层包之外运行，就无法确定相对路径，因此会出现错误。


方法一：使用绝对导入

相对于相对导入，绝对导入更为简洁且易于理解。我们可以使用绝对导入来替代相对导入，该方法不会受到顶层包的限制。 假设我们的项目结构如下：





### 结构 解决
目录结构

    project/
        ├── extensions/
        │  └── sd-iclight/
        │       └── libiclight/
        │           └── monkey_patch.py   
        └── modules/
            └── devices.py

我想在monkey_patch.py中 import devices

    我想在monkey_patch.py中 import devices，文件关系如下：
    project目录下有extensions 目录和modules 目录。
    extensions 目录下有一个子目录 sd-iclight。在 sd-iclight 目录下，有一个名为 libiclight 的子目录。在 libiclight 目录下，里面有一个文件叫做 monkey_patch.py。
    modules 目录下有一个文件 devices.py

from ..modules import devices

    project/
    │
    ├── extensions/
    │   └── sd-iclight/
    │       └── libiclight/
    │           ├── __init__.py
    │           └── monkey_patch.py
    │
    └── modules/
        └── devices.py

在 Python 中，顶层包是指包结构的最上层的包。例如，假设项目结构如下：

    myproject/
        mypackage/
            __init__.py
            module1.py
            subpackage/
                __init__.py
                module2.py


在这个例子中，mypackage 是顶层包，因为它是项目中第一个包含 __init__.py 的目录。subpackage 是 mypackage 的子包。

相对导入是指使用点（.）来导入同一包中的模块或子包。例如，在 module2.py 中，我们可以用以下方式导入 module1 中的内容：

from ..module1 import some_function

这里，.. 表示回到上一层目录，即 mypackage。


from modules import devices




### gpt





假设您的项目结构如下：

    
    myproject/
        mypackage/
            __init__.py
            module1.py
            subpackage/
                __init__.py
                module2.py
        script.py

module1.py：


    def some_function():
        print("Hello from module1!")
module2.py：


    # 错误示例：从顶层包之外导入
    # from ..module1 import some_function

    # 正确示例：使用绝对导入
    from mypackage.module1 import some_function

    def call_function():
        some_function()

script.py：


    # 错误示例：相对导入超出顶层包
    # from .mypackage.module1 import some_function

    # 正确示例：使用绝对导入
    from mypackage.module1 import some_function

    def main():
        some_function()

    if __name__ == "__main__":
        main()


## import
以下两个有区别吗

    import modules.shared as shared
    from modules import shared

避免命名冲突：

使用别名可以避免在同一个文件中导入多个同名模块时的冲突。   
举个例子，如果你在一个文件中需要从不同的包中导入同名模块shared，你可以使用别名来区分它们：

    python
    复制代码
    import package1.shared as shared1
    import package2.shared as shared2

    shared1.some_function()
    shared2.some_function()
而使用from ... import ...的方式则无法处理这种情况：

    python
    复制代码
    from package1 import shared
    from package2 import shared  # 会导致命名冲突
综上所述，这两个导入语句在用法和适用场景上有些细微的区别，选择哪种方式取决于具体需求和代码结构。


# 实例方法修改

## 1

    class ICLightStableDiffusionProcessingImg2Img(processing.StableDiffusionProcessingImg2Img):

        def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            print("This is the new method")

创建有self

## 2

    # 创建原来的实例
    instance = ICLightStableDiffusionProcessingImg2Img()

    # 创建新的实例或定义新的 sample 方法
    new_instance = ICLightStableDiffusionProcessingImg2Img()  # 或者定义新的方法

    # 替换原来实例的 sample 方法
    p.sample = new_instance.sample  # 注意：不要调用方法，只需传递方法的引用

    # 现在 p.sample 包含了新的 sample 方法，可以随后调用它
    p.sample()












#  lambda 函数

lambda arguments: expression

其中，arguments 是参数列表，可以包含零个或多个参数，而 expression 则是函数体，定义了函数的计算逻辑。

Lambda函数通常用于传递给高阶函数，例如 map()、filter()、reduce() 等，或者在需要简短的函数定义时使用

lambda x, y: x + y     

这个Lambda函数有两个参数 x 和 y，它们的输入值将用于执行表达式 x + y 的计算。输入两个参数，Lambda函数将返回它们的和作为输出。


Lambda函数的输入和输出可以是任何类型，取决于参数列表和表达式的定义。你可以在Lambda函数中进行各种计算，包括数学运算、条件判断等，以便根据输入生成输出。



samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

没有输入的lambda



# list 列表 extend append
    results = [[0,0],[1,1],[2,2]]


    final_results = []  
    final_results2 = []  
    for result in results:  
        final_results.extend(result)  
        print(result)
        final_results2.append(result)  

    print(final_results)
    print(final_results2)


    [0, 0]
    [1, 1]
    [2, 2]
    [0, 0, 1, 1, 2, 2]
    [[0, 0], [1, 1], [2, 2]]


## list create

    lista = ["a","b", "c"]
    listb = [lista[:2],lista[2:]]
    print(lista)
    print(listb)


    ['a', 'b', 'c']
    [['a', 'b'], ['c']]



    lista = [("a","b"), ("c",1)]
    listb = [("d","e")]
    print(lista.extend(listb))
    print(listb + lista)



    None
    [('d', 'e'), ('a', 'b'), ('c', 1), ('d', 'e')]




	for index, row in sampled_subset.iterrows():  
		# 这里假设'img'是图像文件的路径或标识符  
		# 你可能需要根据实际情况来调整保存图像的逻辑  
		clean_data.extend((row['imgnames'],label))

这里写进去有误 非元组


	listaa = []
	lista = [("a","b"), ("c",1)]
	listb = [("d","e")]
	print(lista.extend(listb))
	print(listb + lista)
	print(lista.extend(("d","e")))
	print(lista.extend([("d","e")]))


	None
	[('d', 'e'), ('a', 'b'), ('c', 1), ('d', 'e')]
	None
	None

	listaa = []
	listbb = []
	lista = [("a","b"), ("c",1)]
	listb = [("d","e")]
	print(lista.extend(listb))
	print(listb + lista)

	listaa.extend(("d","e"))
	listbb.extend([("d","e")])
	print(listaa)
	print(listbb)
	print(a)

	None
	[('d', 'e'), ('a', 'b'), ('c', 1), ('d', 'e')]
	['d', 'e']
	[('d', 'e')]

	listaa = []
	listbb = []
	lista = [("a","b"), ("c",1)]
	listb = [("d","e")]
	print(lista.extend(listb))
	print(listb + lista)

	print(listaa.extend(("d","e")))
	print(listbb.extend([("d","e")]))

	None
	[('d', 'e'), ('a', 'b'), ('c', 1), ('d', 'e')]
	None
	None

extend竟然不是原地操作返回自身 而是返回一个值！！！！！！！


data1.append((image_info,label)) 可以














## 切片极快 列表遍历极慢 set方法没试过
    val_part0 = img_list_0[i*pernum : (i+1)*pernum]
    val_part1 = img_list_1[i*pernum : (i+1)*pernum]

    #train_part0 = [x for x in img_list_0 if x not in val_part0]  
    #train_part1 = [x for x in img_list_1 if x not in val_part1]  
    train_part0_l = img_list_0[:i*pernum]
    train_part0_r = img_list_0[(i+1)*pernum:]

    train_part1_l = img_list_1[:i*pernum]
    train_part1_r = img_list_1[(i+1)*pernum:]

    train_part1=[]
    train_part1.extend(train_part0_l)
    train_part1.extend(train_part0_r)
    train_part1.extend(train_part1_l)
    train_part1.extend(train_part1_r)


    val_part1.extend(val_part0)










# 字典键值对

## 删除值
字典如何删除某个键和他的键值对


使用del语句
del语句可以直接删除字典中的某个键及其键值对。如果尝试删除的键不存在，则会引发KeyError异常。


    my_dict = {'a': 1, 'b': 2, 'c': 3}  
    
    # 删除键 'b'  
    del my_dict['b']  
    
    print(my_dict)  # 输出: {'a': 1, 'c': 3}


使用pop()方法    
pop()方法用于删除字典中的指定键及其对应的值，并返回该值。如果尝试删除的键不存在，则可以指定一个默认值作为返回值，以避免KeyError异常。


    my_dict = {'a': 1, 'b': 2, 'c': 3}  
    
    # 删除键 'b' 并获取其值  
    value = my_dict.pop('b')  
    print(value)  # 输出: 2  
    print(my_dict)  # 输出: {'a': 1, 'c': 3}  
    
    # 如果尝试删除的键不存在，可以指定一个默认值  
    value = my_dict.pop('d', '键不存在')  
    print(value)  # 输出: 键不存在  
    print(my_dict)  # 输出: {'a': 1, 'c': 3}，字典未变





## 读取


cond_scale = extra_args['cond_scale']

cond_scale = extra_args.get('cond_scale', None)  # 如果键不存在,返回 None


## 设置值
如果你想在赋值前检查键是否存在:

    if 'cond_scale' in extra_args:
        extra_args['cond_scale'] = new_value
    else:
        print("'cond_scale' 不在 extra_args 中")

使用 .update() 方法:

    extra_args.update({'cond_scale': new_value})
这种方法在你需要更新多个值时特别有用。

如果你想在键不存在时添加它:

    extra_args.setdefault('cond_scale', new_value)
这个方法会在键不存在时添加键值对，如果键已存在则不会改变原有的值。









我这样定义

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ): 

kwargs是一个字典


kwargs 是一个字典，其中包含传递给函数的关键字参数的键值对。如果你传递了一个名为 x 的关键字参数给 process_before_every_sampling 函数，那么你可以通过 kwargs['x'] 或者 kwargs.get('x') 来获取它的值。所以你可以这样取数据：

x_value = kwargs.get('x')

或者如果你确定 x 参数肯定存在，你也可以直接使用：

x_value = kwargs['x']

但需要注意的是，如果在调用函数时没有传递 x 这个关键字参数，直接使用 kwargs['x'] 会引发 KeyError 异常。因此，最好使用 get() 方法，这样可以避免异常，并且你可以指定一个默认值。


kwargs 是一个字典，包含了传递给函数的关键字参数的键值对。字典的键是参数名，而对于字典的取值通常使用索引或者 get() 方法。

kwargs.x 这样的语法是试图通过属性访问来获取字典中的值，但这是不合法的 Python 语法，因为 kwargs 是一个字典对象，而字典对象没有名为 x 的属性。

需要使用字典的索引操作或者 get() 方法



# torch tensor stack

    @array_function_dispatch(_stack_dispatcher)
    def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
        """
        Join a sequence of arrays along a new axis.

        The ``axis`` parameter specifies the index of the new axis in the
        dimensions of the result. For example, if ``axis=0`` it will be the first
        dimension and if ``axis=-1`` it will be the last dimension.

Examples

    --------
    >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)

    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)

    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.stack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.stack((a, b), axis=-1)
    array([[1, 4],
           [2, 5],
           [3, 6]])


# torch.cuda.ipc_collect()

torch.cuda.empty_cache()    
torch.cuda.ipc_collect()



torch.cuda.ipc_collect() 是 PyTorch 中的一个函数,它用于手动触发 CUDA 内存的垃圾回收。它的作用是释放已经不再使用的 CUDA 内存空间,以便它们可以被重新分配和利用。    
在某些情况下,PyTorch 可能无法及时释放 CUDA 内存,这可能会导致内存泄漏或内存不足的问题。调用 torch.cuda.ipc_collect() 可以强制立即回收这些未使用的内存块,从而避免这些问题。    
然而,需要注意的是,这个函数只释放内部未使用的内存块,不会影响正在使用的 CUDA 内存。此外,频繁地调用此函数可能会导致性能下降,因为它会引入一些额外的开销。


因此,在大多数情况下,您不需要直接调用 torch.cuda.ipc_collect()。PyTorch 会自动管理和回收 CUDA 内存。但是,如果您确实遇到了内存泄漏或内存不足的问题,并且已经尝试了其他方法(如 torch.cuda.empty_cache() 和 torch.cuda.reset_max_memory_cached())但无效,那么您可以考虑调用 torch.cuda.ipc_collect() 来手动触发垃圾回收。

两者的主要区别在于:

empty_cache()会立即释放PyTorch缓存的所有CUDA内存,包括正在使用和暂时不使用的部分。而ipc_collect()只会释放不再被使用的CUDA内存块。    
empty_cache()的作用范围更广,会清空内部和外部缓存。而ipc_collect()仅在内部进行垃圾回收。   
empty_cache()通常用于主动管理内存,在需要时释放资源。而ipc_collect()主要用于处理内存泄漏等异常情况下的内存回收。     

因此,在正常情况下,如果希望立即释放尽可能多的CUDA内存,您应该优先使用torch.cuda.empty_cache()。而如果怀疑发生了内存泄漏,可以尝试调用torch.cuda.ipc_collect()来回收未使用的内存块。    
通常情况下,您不需要频繁调用这两个函数,因为PyTorch会自动管理CUDA内存。但在某些情况下,手动调用它们可以帮助您更好地管理和监控CUDA内存的使用情况。




# torch.Tensor.scatter_

pytorch函数还是得看文档 不然就得想办法看底层c实现 目前不知道如何debug那个


    Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor

Writes all values from the tensor src into self at the indices specified in the index tensor. For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.


For a 3-D tensor, self is updated as:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

self, index and src (if it is a Tensor) should all have the same number of dimensions. It is also required that index.size(d) <= src.size(d) for all dimensions d, and that index.size(d) <= self.size(d) for all dimensions d != dim. Note that index and src do not broadcast.



This is the reverse operation of the manner described in gather().

torch.gather

    torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor

Gathers values along an axis specified by dim.

For a 3-D tensor the output is specified by:

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

例子

    import torch

    # 创建一个 3x3 的张量并初始化为零，指定数据类型为 torch.float32
    tensor = torch.zeros(3, 3, dtype=torch.float32)

    # 指定乱序的索引位置
    index = torch.tensor([[2, 1, 0]], dtype=torch.long)

    # 指定要插入的值，指定数据类型为 torch.float32
    src = torch.tensor([[10, 20, 30]], dtype=torch.float32)

    # 沿着第0个维度进行操作
    tensor.scatter_(0, index, src)

    print(tensor)

    tensor([[ 0.,  0., 30.],
            [ 0., 20.,  0.],
            [10.,  0.,  0.]])







# 深拷贝
    import copy

    # 创建一个包含嵌套结构的对象
    original_list = [1, 2, [3, 4], 5]

    # 使用deepcopy进行深拷贝
    deep_copied_list = copy.deepcopy(original_list)

    # 使用copy进行浅拷贝
    shallow_copied_list = copy.copy(original_list)

深拷贝在Python中可以应用于大多数对象类型，包括但不限于：

    列表（List）
    字典（Dictionary）
    集合（Set）
    元组（Tuple）
    自定义对象（Classes）
对于内置的可变对象（例如列表、字典和集合），深拷贝将会递归地复制所有嵌套对象。对于不可变对象（例如元组），深拷贝的行为与浅拷贝相同，因为不可变对象没有嵌套对象。

对于自定义对象，如果对象正确地实现了深拷贝方法（__deepcopy__()），则可以成功进行深拷贝。否则，深拷贝将会失败。


# Python计时
```
import time

# 记录开始时间
start_time = time.time()

# 你的程序代码
# 在这里写下你想要测量运行时间的代码

# 记录结束时间
end_time = time.time()

# 计算运行时间
run_time = end_time - start_time
print("程序运行时间为：", run_time, "秒")


```



# try except


    try:
        # 这里放置可能引发异常的代码
        result = 10 / 0  # 这会引发一个 ZeroDivisionError
    except Exception as e:
        print(f"发生了一个异常: {e}")





    try:
        x = int(input("请输入一个数字: "))
        y = 10 / x
    except:
        print("发生了一个异常!")

    print("这行代码会被执行")

这里的关键点是,除了Python内置的一些特殊异常(如SystemExit、KeyboardInterrupt等)之外,一旦异常被except块捕获并处理,程序就会继续执行except块之后的代码。    
无论异常的类型是什么,只要被except块捕获,except块后面的代码都会继续执行。这就是为什么在您的例子中,print("这行代码不会被执行")会被执行。

不需要finally    


    try:
        x = int(input("请输入一个数字: "))
        y = 10 / x
    except ZeroDivisionError:
        print("除数不能为0!")

    print("这行代码不会被执行")
在上面的例子中,如果用户输入一个非数字字符,会引发ValueError异常,但是这个异常没有被捕获,因此程序会崩溃,print("这行代码不会被执行")就不会被执行了。

因此,except块之后的代码是否会执行,取决于是否发生了异常,以及异常是否被成功捕获和处理了。如果异常被适当地捕获和处理,后面的代码就会继续执行;如果异常没有被捕获,程序就会崩溃,后面的代码不会被执行

    try:
        x = int(input("请输入一个数字: "))
        y = 10 / x
        print(f"结果: {y}")
    except ValueError:
        print("您输入的不是一个有效数字!")
    except ZeroDivisionError:
        print("除数不能为0!")
    else:
        print("运算成功!")
    finally:
        print("不管是否发生异常,都会执行这里的代码。")

在上面的示例中,如果用户输入一个非数字字符,将会引发ValueError异常;如果用户输入0,将会引发ZeroDivisionError异常。相应的except语句块将会捕获并处理这些异常。如果没有发生异常,则会执行else语句块中的代码。无论是否发生异常,finally语句块中的代码都会被执行。

    try:
        # 一些可能引发异常的代码
        pass
    except Exception1:
        # 如果发生Exception1类型的异常,执行这里的代码
        pass
    except Exception2:
        # 如果发生Exception2类型的异常,执行这里的代码 
        pass
    else:
        # 如果没有异常发生,执行这里的代码
        pass
    finally:
        # 无论异常是否发生,都会执行这里的代码
        pass

else语句块是可选的,如果try语句块中的代码没有引发任何异常,就会执行else语句块中的代码。


## ctrl c
默认情况不会被捕获   
可以通过 except KeyboardInterrupt捕获












# self

是的,在Python中,您可以使用任何有效的变量名来替代self或cls。Python本身并不限制您必须使用self或cls,这只是一种常见约定和最佳实践。




    class MyClass:
        def method(a, value):
            # 在这里,a代表类本身(类方法)
            print(f"Class value: {value}")

        def other_method(aaa, x, y):
            # 在这里,aaa代表实例本身(实例方法)
            print(f"Instance value: {x + y}")

        @classmethod
        def class_method(bb, value):
            # 在这里,bb代表类本身(类方法)
            print(f"Class value: {value}")


但是,使用self和cls作为变量名有以下优点:

遵循Python的命名约定,增强代码的可读性。
在大型项目中,使用这些约定可以减少混淆,并与其他Python开发人员保持一致。
一些代码检查工具和lint工具可能会对不遵循约定的代码发出警告。

因此,尽管您可以使用其他变量名,但还是建议您坚持使用self和cls,这是Python社区中的标准做法。


# python - 参数

-u: 强制 Python 使用无缓冲的标准输出和标准错误流。这在某些情况下很有用,例如在管道中运行 Python 脚本时。when running Python scripts in a pipeline
-m: 运行一个模块作为脚本。这允许您直接从命令行执行 Python 模块。This allows you to execute Python modules directly from the command line.  Runs a module as a script.
-c: 运行一个命令。这允许您直接在命令行中执行 Python 代码,而不需要单独的脚本文件。

python3 -c "from rembg import remove, new_session; from PIL import Image; output = remove(Image.open('i.png'), session=new_session('u2net', ['CUDAExecutionProvider'])); output.save('o.png')"

更像是c代码



# @机制

## 装饰器（decorators）
@符号用于装饰器（decorators）。装饰器是一种函数或方法，它允许在不改变原函数或方法的情况下，添加额外的功能或修改行为。


装饰器的基本概念
装饰器本质上是一个函数，它接受一个函数作为参数，并返回一个新的函数。装饰器通常用于以下几种用途：

**日志记录（Logging）：**记录函数调用的日志。
**性能测量（Timing）：**测量函数执行时间。
**访问控制与权限验证（Access Control）：**检查用户是否有权限执行某些操作。
**缓存（Caching）：**缓存函数的返回结果以提高性能。


    基本的装饰器示例
    python
    复制代码
    def my_decorator(func):
        def wrapper(*args, **kwargs):
            print("Something is happening before the function is called.")
            result = func(*args, **kwargs)
            print("Something is happening after the function is called.")
            return result
        return wrapper

    @my_decorator
    def say_hello():
        print("Hello!")

    say_hello()

但是我遇到的wrapper不是这样用 @也有用在别的位置 如在类创建时自动调用

这应该就属于引用函数前先调用@




在这个例子中，my_decorator是一个装饰器，say_hello函数被装饰器包装。执行say_hello()时，会先打印装饰器中的前置和后置消息。

如何定义自己的装饰器
定义装饰器的一般步骤如下：

    定义一个装饰器函数，它接受一个函数作为参数。
    在装饰器函数内部，定义一个嵌套的包装器函数（wrapper），它调用原函数并添加额外的行为。
    包装器函数返回装饰器函数的结果。
    在原函数上应用装饰器。


    import time

    def timer_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete")
            return result
        return wrapper

    @timer_decorator
    def some_function():
        time.sleep(2)
        print("Function is running")

    some_function()
在这个示例中，timer_decorator是一个装饰器，用于测量some_function函数的执行时间。

多个装饰器
一个函数可以同时被多个装饰器修饰。装饰器的应用顺序是自下而上，即最上面的装饰器最后被应用。下面是一个示例：

    python
    复制代码
    def decorator1(func):
        def wrapper(*args, **kwargs):
            print("Decorator 1")
            return func(*args, **kwargs)
        return wrapper

    def decorator2(func):
        def wrapper(*args, **kwargs):
            print("Decorator 2")
            return func(*args, **kwargs)
        return wrapper

    @decorator1
    @decorator2
    def say_hello():
        print("Hello!")

    say_hello()
    输出结果是：

    复制代码
    Decorator 1
    Decorator 2
    Hello!
在这个例子中，decorator2首先修饰say_hello函数，然后decorator1再修饰decorator2的返回函数。


## 除了装饰器
@符号在Python中还可以用于其他位置，特别是在类的定义中，它可以用于声明类方法（class methods）、静态方法（static methods）、以及属性（properties）。


类方法（@classmethod）

类方法是绑定到类而不是实例的方法。它的第一个参数是类本身，通常命名为cls。类方法通常用于创建工厂方法（factory methods），即返回类的实例的方法。

    python
    复制代码
    class MyClass:
        class_variable = "class variable"

        @classmethod
        def class_method(cls):
            print(f"This is a class method. {cls.class_variable}")

    MyClass.class_method()  # 调用类方法

看不出有什么特殊


静态方法（@staticmethod）

静态方法不绑定到实例或类，它们类似于普通函数，只是定义在类的命名空间中。静态方法不能访问或修改类状态。

    python
    复制代码
    class MyClass:
        @staticmethod
        def static_method():
            print("This is a static method.")

    MyClass.static_method()  # 调用静态方法


属性（Properties）

@property装饰器用于将方法转换为属性，使得可以通过点操作符访问方法。

    python
    复制代码
    class MyClass:
        def __init__(self, value):
            self._value = value

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, new_value):
            if new_value < 0:
                raise ValueError("Value cannot be negative")
            self._value = new_value

    obj = MyClass(10)
    print(obj.value)  # 访问属性
    obj.value = 20  # 设置属性
    print(obj.value)
在这个例子中，value方法被@property装饰器转换为一个属性，可以通过obj.value进行访问和设置。

装饰器： 使用@来修饰函数或方法，添加额外的功能。
类方法： 使用@classmethod来定义，绑定到类本身而不是实例。
静态方法： 使用@staticmethod来定义，不绑定到实例或类，类似于普通函数。
属性： 使用@property来定义，使得方法可以像属性一样访问。


### validator
Pydantic是一个用于数据校验和设置的库，广泛用于FastAPI等框架中。@validator装饰器用于在字段赋值之前或之后进行额外的验证或处理。

root_validator的数据校验有点特别     
甚至用了模型    
甚至可以数据校验中完成所有操作，然后直接返回



这个应该是之前

    @validator("input_fg", "uploaded_bg", pre=True, allow_reuse=True)
    def parse_image(cls, value) -> np.ndarray:
        if isinstance(value, str):
            return cls.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or value is None
        return value

这个装饰器接受多个参数：

    **字段名（如"input_fg", "uploaded_bg"）：**指定要验证的字段，可以是一个或多个字段名。
    **pre参数：**如果设置为True，则在字段赋值之前执行验证逻辑，否则在字段赋值之后执行。
    **allow_reuse参数：**如果设置为True，则允许重复使用相同的验证函数在多个字段上。


这个装饰器应用于parse_image方法，用于验证和处理input_fg和uploaded_bg这两个字段。

示例作用    
这个示例的主要作用是确保input_fg和uploaded_bg字段在赋值前是合适的数据类型。如果字段值是字符串，它会将其解码为np.ndarray。如果字段值已经是np.ndarray或None，则直接返回该值。

应用场景    
这种用法在处理输入数据时非常有用，特别是当需要对数据进行预处理或验证时。举例如下：

    class ImageModel(BaseModel):
        input_fg: np.ndarray
        uploaded_bg: np.ndarray

        @validator("input_fg", "uploaded_bg", pre=True, allow_reuse=True)
        def parse_image(cls, value) -> np.ndarray:
            if isinstance(value, str):
                return cls.cls_decode_base64(value)
            assert isinstance(value, np.ndarray) or value is None
            return value

        @classmethod
        def cls_decode_base64(cls, value: str) -> np.ndarray:
            # 这里应该有解码逻辑，将base64字符串转换为np.ndarray
            pass

    # 使用示例
    image_model = ImageModel(input_fg="some_base64_string", uploaded_bg=None)
在这个示例中，ImageModel类定义了两个字段input_fg和uploaded_bg。通过使用@validator装饰器，可以在字段赋值前对其进行处理和验证，确保数据类型的正确性。




    class ICLightArgs(BaseModel):
        enabled: bool = False
        model_type: ModelType = ModelType.FC
        input_fg: Optional[np.ndarray] = None
        uploaded_bg: Optional[np.ndarray] = None
        bg_source_fc: BGSourceFC = BGSourceFC.NONE
        bg_source_fbc: BGSourceFBC = BGSourceFBC.UPLOAD
        remove_bg: bool = True

        # FC model only option. Overlay the FG image on top of the light map
        # in order to better preserve FG's base color.
        reinforce_fg: bool = True
        # Transfer high frequency detail from input image to the output.
        # This can better preserve the details such as text.
        detail_transfer: bool = False
        # Whether to use raw input for detail transfer.
        detail_transfer_use_raw_input: bool = False
        # Blur radius for detail transfer.
        detail_transfer_blur_radius: int = 5

        # Calculated value of the input fg with alpha channel filled with grey.
        input_fg_rgb: Optional[np.ndarray] = None

调用位置

    @classmethod
    def fetch_from(cls, p: StableDiffusionProcessing):
        script_runner: scripts.ScriptRunner = p.scripts
        ic_light_script: scripts.Script = [
            script
            for script in script_runner.alwayson_scripts
            if script.title() == "IC Light"
        ][0]
        args = p.script_args[ic_light_script.args_from : ic_light_script.args_to]
        assert len(args) == 1
        try:
            return ICLightArgs(**args[0])

        从这输入参数    
        并使用validator重新处理参数      


@validator(pre=True)
执行时机： 在字段赋值之前执行。
作用范围： 作用于单个字段或多个指定的字段。
典型用途： 对单个字段进行预处理和验证，例如类型转换、格式校验等。

@root_validator
执行时机： 在所有字段的验证之后执行，可以控制在字段赋值之前（pre=True）或之后（默认）。
作用范围： 作用于整个模型，可以访问和处理所有字段。
典型用途： 对整个模型的数据进行综合验证，例如跨字段的逻辑验证。


    from pydantic import BaseModel, validator, root_validator
    import numpy as np

    class MyModel(BaseModel):
        input_fg: np.ndarray
        uploaded_bg: np.ndarray

        @validator("input_fg", "uploaded_bg", pre=True, allow_reuse=True)
        def pre_validator(cls, value):
            print("Executing @validator(pre=True)")
            if isinstance(value, str):
                return cls.decode_base64(value)
            assert isinstance(value, np.ndarray) or value is None
            return value

        @root_validator(pre=True)
        def pre_root_validator(cls, values):
            print("Executing @root_validator(pre=True)")
            # 添加逻辑，例如处理互相依赖的字段
            return values

        @root_validator
        def post_root_validator(cls, values):
            print("Executing @root_validator")
            # 添加逻辑，例如处理互相依赖的字段
            return values

        @validator("input_fg", "uploaded_bg", allow_reuse=True)
        def post_validator(cls, value):
            print("Executing @validator")
            # 添加逻辑
            return value

        @classmethod
        def decode_base64(cls, value: str) -> np.ndarray:
            # 实现base64解码逻辑，这里假设解码后是一个空数组
            return np.array([])

    # 使用示例
    data = {
        "input_fg": "some_base64_string",
        "uploaded_bg": None
    }

    model = MyModel(**data)
    输出结果
    当你运行这个代码时，输出结果是：

    graphql
    复制代码
    Executing @validator(pre=True)
    Executing @validator(pre=True)
    Executing @root_validator(pre=True)
    Executing @root_validator
    Executing @validator
    Executing @validator

顺序

    @validator(pre=True)：在字段赋值之前，按定义顺序执行所有@validator(pre=True)装饰器。
    字段赋值：字段赋值并进行常规验证（类型检查等）。
    @root_validator(pre=True)：在字段赋值之后，按定义顺序执行所有@root_validator(pre=True)装饰器。
    @root_validator：在字段赋值和常规验证完成后，按定义顺序执行所有@root_validator装饰器。
    @validator：在字段赋值和常规验证完成后，按定义顺序执行所有@validator装饰器。


validator 感觉是 直接定义在 赋值的函数中 在赋值前完成先调用     






# 回调 括号外括号的方法

这份代码没有直接用 @   
以及 @ 然后括号传参    
@app.post('/vid2vid_infer')     
甚至写成一个类 一个方法    
或者再声明成属性


为什么在使用时hook_close跟着两个括号

    def hook_close(patcher_field_name: str):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_close_func(self: StableDiffusionProcessing, *args, **kwargs):
                patcher: ModelPatcher = getattr(self, patcher_field_name)
                # 主要想在这里使用内部的 patcher_field_name 并防止互相冲突
                assert isinstance(patcher, ModelPatcher)
                patcher.close()
                logger.info(f"Close p.{patcher_field_name}.")
                return func(self, *args, **kwargs)

            return wrapped_close_func

        return decorator

    StableDiffusionProcessingTxt2Img.close = hook_close("model_patcher")(
        StableDiffusionProcessingTxt2Img.close
    )


在使用 hook_close 时跟着两个括号的原因如下：

    第一个括号 hook_close("model_patcher") 调用高阶函数 hook_close，返回一个装饰器函数 decorator。
    第二个括号 decorator(StableDiffusionProcessingTxt2Img.close) 调用装饰器函数，将原始的 close 方法传入，返回一个新的包装函数。
最终，这段代码为 StableDiffusionProcessingTxt2Img.close 方法增加了额外的关闭 patcher 和记录日志的功能。

hook_close 是一个接受字符串参数 patcher_field_name 的高阶函数。

functools.wraps 用于保持原始函数的元数据。





双包装 。 
是否有必要？？？      
想要多传一个参数 所以双包装？


    def hook_sample():
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_sample_func(self: Sampler, *args, **kwargs):
                patcher: ModelPatcher = self.p.get_model_patcher()
                assert isinstance(patcher, ModelPatcher)
                patcher.patch_model()
                logger.info(f"Patch {patcher.name}.")

                try:
                    return func(self, *args, **kwargs)
                finally:
                    patcher.unpatch_model()
                    logger.info(f"Unpatch {patcher.name}.")

            return wrapped_sample_func

        return decorator

    Sampler.launch_sampling = hook_sample()(Sampler.launch_sampling)

这个直接就是 调用时 定义一个函数 然后返回这个新函数    
有必要吗？   
这个不如直接用    

到是上面的可能有必要     
规整一些    
传入关键字     


@functools.wraps(func) 是 Python 的 functools 模块提供的一个装饰器，它主要用于装饰器函数内部，以确保被装饰函数的元数据（例如函数名、文档字符串等）被保留。这在调试和文档生成时非常有用。


保留原始函数的名称和文档字符串：装饰器通常会创建一个新的包装函数，这会导致原始函数的名称和文档字符串被覆盖。@functools.wraps(func) 确保这些元数据被保留，从而使包装后的函数仍然能够显示原始函数的信息。

保持函数的签名：@functools.wraps(func) 可以使包装函数的签名看起来与原始函数相同。这对于自动化工具（如 IDE、调试器和文档生成器）非常重要，因为这些工具依赖于函数签名来提供有用的信息和功能。

保留其他元数据：除了函数名和文档字符串，@functools.wraps(func) 还会保留原始函数的模块和注解等其他元数据。

    import functools

    def my_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function"""
            print("Something is happening before the function is called.")
            result = func(*args, **kwargs)
            print("Something is happening after the function is called.")
            return result
        return wrapper

    @my_decorator
    def say_hello(name):
        """Greet someone by their name"""
        return f"Hello, {name}!"

    print(say_hello.__name__)  # 输出: say_hello
    print(say_hello.__doc__)   # 输出: Greet someone by their name



装饰后的 say_hello 函数仍然保留了原始函数的名称和文档字符串。    
如果没有使用 @functools.wraps(func)，输出会变成：

    python
    复制代码
    print(say_hello.__name__)  # 输出: wrapper
    print(say_hello.__doc__)   # 输出: Wrapper function


这表明装饰后的函数失去了原始函数的名称和文档字符串，这在调试和文档生成时会带来不便。通过使用 @functools.wraps(func)，我们确保这些元数据得以保留。


在编程语言中，特别是在函数式编程和面向对象编程中，函数或方法的签名指的是它的参数类型和返回类型的描述。签名通常包括以下几个方面：

函数名：函数或方法的名称。

参数列表：函数或方法定义的参数类型、顺序和数量。这描述了函数接受的输入。

返回类型：函数或方法的返回值类型。它描述了函数执行后返回的结果的类型。

签名在编程中具有重要的作用，主要体现在以下几个方面：

类型检查和静态分析：编译器或解释器可以使用函数签名来进行类型检查，确保调用者和调用函数的代码能够在类型上正确匹配。这种静态类型检查可以帮助捕获潜在的类型错误，提高代码的健壮性。

文档生成：函数签名是生成文档的重要组成部分，特别是自动生成文档时。文档生成工具使用签名来显示函数的参数和返回值信息，帮助开发者理解如何正确使用函数。

自动化工具支持：集成开发环境（IDE）、调试器和其他开发工具通常使用函数签名来提供智能补全、代码导航和调试信息，帮助开发者更高效地编写和调试代码。



    def greet(name: str) -> str:
        return f"Hello, {name}!"

    # 函数签名：
    # 函数名: greet
    # 参数列表: name: str
    # 返回类型: str


在这个例子中，greet 函数的签名包括函数名 greet、参数列表 name: str 和返回类型 str。这个签名告诉我们，greet 函数接受一个字符串类型的参数 name，并返回一个字符串类型的结果。

总结     
函数或方法的签名是其定义的重要组成部分，描述了其名称、参数类型和返回类型。签名在静态类型检查、文档生成和开发工具支持等方面都起到了关键作用，帮助开发者编写和维护高质量的代码。

在 Python 编程中，函数或方法可以具有额外的元数据，其中包括模块和注解。这些元数据可以提供有关函数或方法更多的信息，使得代码更具表现力和可理解性。

1. 模块（Module）    
在函数或方法的上下文中，模块指的是包含该函数或方法的 Python 模块（即 .py 文件）。在 Python 中，每个 .py 文件都是一个模块，模块由全局变量、函数和类组成。函数或方法的模块信息表示它所属的 Python 模块，这对于跟踪函数定义的来源和组织代码非常有用

示例：
假设有一个名为 utils.py 的文件，其中定义了一个函数 process_data：

    python
    复制代码
    # utils.py

    def process_data(data):
        # function body
        pass
    在其他地方调用 process_data 函数时，可以通过函数的 __module__ 属性获取它所在的模块：

    python
    复制代码
    import utils

    print(utils.process_data.__module__)  # 输出: utils
这里 utils.process_data.__module__ 的值是 'utils'，指示该函数定义在 utils.py 模块中。


注解（Annotations）    
Python 3 中引入了函数注解的概念，它允许程序员在函数定义中添加对参数和返回值的注释信息。这些注释信息不会影响函数的运行方式，但可以提供关于参数类型、预期参数值或返回值类型的提示和文档。


    def greet(name: str) -> str:
        return f"Hello, {name}!"
在这个示例中，name: str 和 -> str 就是函数 greet 的注解：

name: str 表示函数接受一个名为 name 的参数，并且该参数的预期类型为字符串 (str)。
-> str 表示函数返回一个字符串 (str) 类型的值。
这些注解可以通过函数对象的 __annotations__ 属性来访问：

    python
    复制代码
    print(greet.__annotations__)  # 输出: {'name': <class 'str'>, 'return': <class 'str'>}

greet.__annotations__ 的值是一个字典，包含了参数和返回值的注解信息。


模块：函数或方法的模块元数据表示它所属的 Python 模块。    
注解：函数或方法的注解提供了关于参数和返回值类型的描述信息，用于文档生成和代码理解。    
这些元数据提供了额外的信息，帮助开发者理解和使用函数或方法，以及维护高质量的代码。

在这个示例中，: str 和 -> str 就是函数 greet 的注解。name: str 表示参数 name 的类型为 str，-> str 表示函数返回值的类型为 str。


signature = inspect.signature(greet)
print(signature)  # 输出: (name: str) -> str
在这个示例中，inspect.signature(greet) 返回了 greet 函数的签名对象，其结果为 (name: str) -> str，包含了函数的参数和返回值类型信息。


注解提供了一种灵活的方式来为函数参数和返回值添加元数据，这些信息可以用于文档生成、类型检查等。它们是一种描述性的元数据，而不直接影响函数的执行。

签名则更加具体和实际，直接描述了函数的输入和输出方式，是函数定义的核心部分。签名通常是由函数的名称、参数列表和返回类型组成，用于指导函数的调用和使用。

因此，虽然注解可以提供签名中的某些信息（如参数类型和返回类型），但注解本身更多地是一种元数据的扩展，用于提供更多的上下文和文档支持。

Python 中的函数注解是一种语法特性，允许开发者在函数定义中为参数和返回值添加额外的元数据信息，这些信息可以是任何对象，通常用来指定参数的类型或者提供其他辅助信息。函数注解并不会影响函数的行为或返回值，它们只是提供了一种标记方法来帮助开发者和工具理解函数的用途和预期用法。

签名（Signature）
函数的签名指的是函数定义的一部分，主要包括函数的名称、参数列表和返回类型。签名反映了函数如何接受输入和产生输出，是函数的核心定义部分。在 Python 中，函数签名可以通过 inspect 模块中的 signature 对象来访问和操作。

示例：

    python
    复制代码
    import inspect

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    signature = inspect.signature(greet)
    print(signature)  # 输出: (name: str) -> str


## @app.post('/vid2vid_infer')

    from fastapi import FastAPI, Request
    from functools import wraps
    import random
    import threading
    import logging

    app = FastAPI()
    logger = logging.getLogger(__name__)

    # 自定义装饰器
    def post_with_params(path: str):
        def decorator(func):
            @app.post(path)
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    # 示例数据类
    class InferenceRequest:
        def __init__(self, ai_pic_id, seed=None, return_info="", batch_name="", ckpt_name=""):
            self.ai_pic_id = ai_pic_id
            self.seed = seed
            self.return_info = return_info
            self.batch_name = batch_name
            self.ckpt_name = ckpt_name

    # 示例处理函数
    @post_with_params('/vid2vid_infer')
    async def vid2vid_infer(request: InferenceRequest):
        if request.seed is None or request.seed == -1:
            request.seed = random.randint(1, 2 ** 32)
        
        logger.info(f'NEW REQUEST {request.ai_pic_id}\n{request}')
        status = 0
        msg = 'success'
        
        if vid2vid.busy:
            status = 1  # 示例错误码
            msg = f'Inference in the thread, model_id={request.return_info}, busy={vid2vid.busy}'
            logger.info(msg)
            return {'code': 200, 'batch_name': request.batch_name, 'status': status, 'msg': msg}
    
        thread_one = threading.Thread(target=video2video, args=(request,))
        thread_one.start()
        
        logger.info(f'send back batch_name: {request.batch_name}')
        return {'code': 200, "base_model": request.ckpt_name, 'batch_name': request.batch_name, 'status': status, 'msg': msg}

    # 示例函数，用于模拟视频处理
    def video2video(request):
        pass

    # 示例对象，用于模拟系统状态
    class Vid2Vid:
        busy = False

    vid2vid = Vid2Vid()


定义装饰器post_with_params：

接受一个路径参数path。
内部定义一个装饰器函数decorator，接受一个函数func。
使用@app.post(path)将func绑定到指定的路径，处理POST请求。
返回包装函数wrapper，确保原始函数的功能保持不变。
使用装饰器：

使用@post_with_params('/vid2vid_infer')装饰vid2vid_infer函数，将其绑定到/vid2vid_infer路径。


### 自定义

要实现一个自定义装饰器，能够像 @lujunda.post('/vid2vid_infer') 这样的用法，你可以创建一个类来处理装饰器，并使用内部函数来模拟请求处理的行为。下面是一个示例，展示如何实现这个功能。


    from functools import wraps
    import random
    import threading
    import logging

    # 设置日志记录器
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # 示例系统状态对象
    class Vid2Vid:
        busy = False

    vid2vid = Vid2Vid()

    # 示例任务栈
    work_stack = []

    # 示例错误代码
    ERR_ALG = 1

    # 模拟一个请求数据类
    class InferenceRequest:
        def __init__(self, ai_pic_id, seed=None, return_info="", batch_name="", ckpt_name=""):
            self.ai_pic_id = ai_pic_id
            self.seed = seed
            self.return_info = return_info
            self.batch_name = batch_name
            self.ckpt_name = ckpt_name

自定义装饰器类

    class Lujunda:
        def __init__(self):
            self.routes = {}

        def post(self, path: str):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    logger.info(f'Handling POST request at {path}')
                    return func(*args, **kwargs)
                self.routes[path] = wrapper
                return wrapper
            return decorator

    lujunda = Lujunda()


    # 示例处理函数
    @lujunda.post('/vid2vid_infer')
    def vid2vid_infer(request: InferenceRequest):
        # 设置随机种子
        if request.seed is None or request.seed == -1:
            request.seed = random.randint(1, 2 ** 32)
        
        # 记录请求信息
        logger.info(f'NEW REQUEST {request.ai_pic_id}\n{request}')
        status = 0
        msg = 'success'
        
        # 检查系统状态
        if vid2vid.busy:
            status = ERR_ALG
            msg = f'Inference in the thread, model_id={request.return_info}, busy={vid2vid.busy}'
            logger.info(msg)
            return {'code': 200, 'batch_name': request.batch_name, 'status': status, 'msg': msg}
    
        # 启动新线程处理任务
        thread_one = threading.Thread(target=video2video, args=(request,))
        thread_one.start()
        
        # 返回响应
        logger.info(f'send back batch_name: {request.batch_name}')
        return {'code': 1, "base_model": request.ckpt_name, 'batch_name': request.batch_name, 'status': status, 'msg': msg}

    # 示例函数，用于模拟视频处理
    def video2video(request):
        logger.info(f"Processing video for request {request.ai_pic_id}")

    # 模拟发送POST请求
    def simulate_post_request():
        path = '/vid2vid_infer'
        request = InferenceRequest(ai_pic_id="12345", batch_name="batch1", ckpt_name="model_v1")
        if path in lujunda.routes:
            response = lujunda.routes[path](request)
            print(response)
        else:
            print(f"No handler for path: {path}")

    # 运行模拟POST请求
    simulate_post_request()



定义装饰器类 Lujunda：

这个类用于存储路径和处理函数的映射关系。
post 方法接受路径参数 path 并返回一个装饰器 decorator。
decorator 内部定义一个包装函数 wrapper，在调用原始函数之前记录请求路径信息。
wrapper 被存储在 self.routes 字典中，键为路径，值为包装函数。
使用装饰器：

使用 @lujunda.post('/vid2vid_infer') 装饰 vid2vid_infer 函数，将其绑定到指定的路径，并处理模拟的POST请求。

模拟发送POST请求：

定义 simulate_post_request 函数，创建一个 InferenceRequest 对象并调用 lujunda.routes[path] 处理请求。
打印响应结果。





## **回调（callback）**

通常指的是一个函数或者代码块，它作为参数传递给另一个函数，并且在某些事件发生或者特定条件满足时被调用执行。简单来说，回调函数就是通过将一个函数作为参数传递给另一个函数，从而在某个事件发生时执行这个函数。这种机制常用于异步编程，事件驱动编程或者处理复杂的程序流程。

常见的例子包括处理用户界面事件（比如点击按钮后执行的函数）、处理异步操作（比如文件读取完成后执行的函数）、或者处理网络请求（比如收到服务器响应后执行的函数）等。

当你在使用 JavaScript 进行异步编程时，回调函数经常会用到。以下是几个常见的例子：

事件处理：

    // HTML 中的按钮
    <button id="myButton">Click Me</button>

    // JavaScript 中的事件处理函数
    document.getElementById('myButton').addEventListener('click', function() {
        alert('Button clicked!');
    });
这里的匿名函数 function() { alert('Button clicked!'); } 就是一个回调函数，它在按钮被点击时被调用执行。


定时器：

    javascript
    复制代码
    // 使用 setTimeout 设置定时器
    setTimeout(function() {
        console.log('Timeout completed!');
    }, 2000);
这里的 function() { console.log('Timeout completed!'); } 就是一个回调函数，它在延时结束后被调用执行。


网络请求（使用 jQuery 的例子）：

    javascript
    复制代码
    // 发起 AJAX 请求
    $.ajax({
        url: '/api/data',
        success: function(response) {
            console.log('Data received:', response);
        },
        error: function(xhr, status, error) {
            console.error('Error occurred:', error);
        }
    });
这里的 success 和 error 参数中的函数都是回调函数，在请求成功或失败时被调用执行。




在 Python 中，回调函数的使用场景也是很常见的，尤其是在事件驱动或者异步编程中。以下是一些 Python 中回调函数的例子：

事件驱动（使用 tkinter GUI 库的例子）：

    python
    复制代码
    import tkinter as tk

    def button_clicked():
        print('Button clicked!')

    root = tk.Tk()
    button = tk.Button(root, text='Click Me', command=button_clicked)
    button.pack()
    root.mainloop()
这里的 button_clicked 函数就是一个回调函数，它在按钮被点击时被调用执行。


异步操作（使用 asyncio 库的例子）：

    python
    复制代码
    import asyncio

    async def main():
        print('Start sleeping')
        await asyncio.sleep(2)
        print('Awake!')

    def callback():
        print('Callback executed')

    loop = asyncio.get_event_loop()
    loop.call_later(1, callback)  # 在1秒后调用回调函数
    loop.run_until_complete(main())
这里的 callback 函数就是一个回调函数，它在指定的时间后被调用执行。


async def main():

定义了一个异步函数 main，它是一个协程（coroutine）。协程是一种特殊的函数，可以在需要时被挂起和恢复，以便异步执行。


loop = asyncio.get_event_loop()

获取当前的事件循环对象。

loop.run_until_complete(main())

运行事件循环，直到 main() 协程完成。这会启动 main() 函数中定义的异步操作，并且会在等待 2 秒后打印 'Awake!'。

运行这段代码后，输出应该是：

    sql
    复制代码
    Start sleeping
    Callback executed
    Awake!
解释每个输出的产生过程：

首先，程序执行 print('Start sleeping')，打印出 'Start sleeping'。
然后，事件循环调度 loop.call_later(1, callback)，在 1 秒后执行 callback 函数。所以，在执行 main() 中的异步操作期间，1 秒钟过去后，callback 函数被调用，打印出 'Callback executed'。
最后，await asyncio.sleep(2) 让 main() 协程挂起，等待 2 秒钟。一旦等待完成，程序继续执行，打印出 'Awake!'。
这种输出顺序说明了异步操作的特性：即使在等待异步操作的过程中，程序可以继续处理其他任务或调用回调函数，而不必阻塞整个进程。









事件处理（使用 Flask Web 框架的例子）：

    python
    复制代码
    from flask import Flask, request

    app = Flask(__name__)

    @app.route('/webhook', methods=['POST'])
    def webhook():
        data = request.json
        handle_webhook(data, callback)
        return 'OK'

    def callback(response):
        print('Received response:', response)

    def handle_webhook(data, callback):
        # 处理 webhook 数据，并调用回调函数
        callback(data)

    if __name__ == '__main__':
        app.run(debug=True)
这里的 callback 函数作为参数传递给 handle_webhook 函数，在处理完 webhook 数据后被调用执行。


回调函数通过将函数作为参数传递，实现了灵活的程序控制和异步操作管理。

异步操作指的是程序在执行过程中不需要等待某些耗时的任务完成，而是可以继续执行后续的代码。这种方式允许程序在等待某些任务完成的同时，可以处理其他任务，从而提高了程序的效率和响应速度。

在编程中，异步操作通常用于处理需要等待的 I/O 操作（如文件读写、网络请求、数据库访问等），以及定时器等需求。Python 中有几种方式可以实现异步操作，比如使用 asyncio 库进行协程的管理，或者使用异步框架如 aiohttp 处理网络请求。

    import asyncio

    async def async_task(name, delay):
        print(f'{name} started')
        await asyncio.sleep(delay)
        print(f'{name} finished after {delay} seconds')

    async def main():
        task1 = async_task('Task 1', 2)
        task2 = async_task('Task 2', 1)
        
        # 并发执行异步任务
        await asyncio.gather(task1, task2)

    if __name__ == '__main__':
        asyncio.run(main())
在这个例子中，async_task 是一个异步函数，模拟了需要等待一段时间后完成的任务。在 main 函数中，通过 asyncio.gather 同时启动了多个异步任务，而不必等待每个任务完成才执行下一个任务。这样可以在一个任务等待时并发执行其他任务，提高了程序的效率和响应能力。

异步操作的体现在于，程序在遇到需要等待的操作时，并不会阻塞整个进程或线程，而是利用事件循环（event loop）机制，让其他任务继续执行，直到等待的操作完成后再返回处理结果。


# log常用

## cmd

python -u gradio_demo.py 2>&1 | tee -a log.txt


## logger

### 灵活设置多个logger
该方法不可用

    def logger_init(outpath):
        log_path = os.path.join(outpath, "run2.log")

        logger = logging.getLogger(__name__)  
        logger.setLevel(logging.INFO)  
        # 如果需要，先移除所有现有的handlers  
        logger.handlers.clear()  
        
        # 添加新的FileHandler  
        handler = logging.FileHandler(log_path, mode='a+')  
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))  
        logger.addHandler(handler)  

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))  
        logger.addHandler(handler)  
        
        return logger

    logger = logger_init("./")


    def logger_init_specific(outpath):  
        logger = logging.getLogger(__name__)  
        logger.setLevel(logging.INFO)  
        # 如果需要，先移除所有现有的handlers  
        logger.handlers.clear()  
        
        # 添加新的FileHandler  
        handler = logging.FileHandler(outpath, mode='w')  
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))  
        logger.addHandler(handler)  
        
        return logger




### 固定一种basicConfig 无法创建多种形式logger 容易混乱
    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)


使用

    logger.info(qnn)

    if not config.quant.grad_checkpoint:
        logger.info('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)

    logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")



用法      
勉强能用 也会混乱

    def logger_init(outpath):
        log_path = os.path.join(outpath, "run2.log")
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        return logger
    logger = logger_init("./")
    def logger_init_specific(outpath):  
        logger = logging.getLogger(__name__)  
        logger.setLevel(logging.INFO)  
        # 如果需要，先移除所有现有的handlers  
        logger.handlers.clear()  
        
        # 添加新的FileHandler  
        handler = logging.FileHandler(outpath, mode='w')  
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))  
        logger.addHandler(handler)  
        
        return logger

### log不能像print使用
logger.info("打乱前的1", data1[:5])


### 全局
    # 在模块级别创建日志记录器  
    _logger = None  
    
    def logger_init(log_file):  
        global _logger  # 这里使用global是可选的，但如果你打算在函数内部修改_logger，就需要它  
        if _logger is None:  
            _logger = logging.getLogger('my_logger')  
            _logger.setLevel(logging.DEBUG)  
            
            # 添加文件处理器  
            fh = logging.FileHandler(log_file)  
            fh.setLevel(logging.DEBUG)  
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
            fh.setFormatter(formatter)  
            _logger.addHandler(fh)  
            
            # 可以选择性地添加控制台处理器  
            ch = logging.StreamHandler()  
            ch.setLevel(logging.ERROR)  
            ch.setFormatter(formatter)  
            _logger.addHandler(ch)  





# yaml文件加载方式


opt = parser.parse_args()

config = OmegaConf.load(f"{opt.config}")

路径给到直接读取

很像是p 或者pipeline的字典



# 类方法 实例方法
类方法如果不创建实例无法使用实例方法

什么这个函数无法进去converted_dict = cls.convert_args_to_dict(args)？
定义是这样的def convert_args_to_dict(self, args) -> dict:

Edit
您提出了一个很好的问题。问题出在方法的定义和调用方式上。让我解释一下：

方法定义:

def convert_args_to_dict(self, args) -> dict:
这个方法被定义为实例方法（instance method），因为它的第一个参数是 self。

调用方式:

converted_dict = cls.convert_args_to_dict(args)
这里您试图通过类（cls）来调用这个方法，而不是通过实例。

这就造成了不匹配。当您通过类调用方法时，Python 期望这个方法是一个类方法（class method），但它实际上被定义为实例方法。

解决这个问题有两种主要方法：

将方法改为类方法: 如果这个方法不需要访问实例的特定属性，您可以将其改为类方法：
python

Copy
@classmethod
def convert_args_to_dict(cls, args) -> dict:
    # 方法内容保持不变
这样，您就可以通过类来调用这个方法，就像您原本尝试的那样。

创建类的实例来调用方法: 如果出于某些原因，这个方法必须保持为实例方法，您需要先创建类的实例，然后通过实例调用这个方法：
python

Copy
instance = cls()
converted_dict = instance.convert_args_to_dict(args)
考虑到您的使用场景（在 fetch_from 类方法中调用），第一种方法（将 convert_args_to_dict 改为类方法）可能更合适。

修改后的 convert_args_to_dict 方法应该是这样的：

python

Copy
@classmethod
def convert_args_to_dict(cls, args) -> dict:
    # 方法内容保持不变
    ...
这样修改后，您原来的调用方式 cls.convert_args_to_dict(args) 就应该可以正常工作了。

这个问题强调了理解 Python 中实例方法、类方法和静态方法之间区别的重要性。在这种情况下，因为您想在类方法 (fetch_from) 中调用 convert_args_to_dict，所以将后者也定义为类方法是最合适的。


def convert_args_to_dict(self, args) -> dict:   
这样就进不去   

必须下面这样

    @classmethod
    def convert_args_to_dict(cls, args) -> dict:
        # 定义参数的顺序
        arg_names = [
            "enabled",
            "model_type",
            "input_fg",
            "uploaded_bg",
            "bg_source_fc",
            "bg_source_fbc",
            "remove_bg",
            "reinforce_fg",
            "detail_transfer",
            "detail_transfer_use_raw_input",
            "detail_transfer_blur_radius",
        ]
        
        # 创建一个字典，只包含提供的参数
        result = {}
        for i, arg in enumerate(args):
            if i < len(arg_names):
                result[arg_names[i]] = arg
        
        # 转换枚举类型
        if "model_type" in result and isinstance(result["model_type"], str):
            result["model_type"] = ModelType(result["model_type"])
        if "bg_source_fc" in result and isinstance(result["bg_source_fc"], str):
            result["bg_source_fc"] = BGSourceFC(result["bg_source_fc"])
        if "bg_source_fbc" in result and isinstance(result["bg_source_fbc"], str):
            result["bg_source_fbc"] = BGSourceFBC(result["bg_source_fbc"])
        
        return result




    @classmethod
    def fetch_from(cls, p: StableDiffusionProcessing):
        script_runner: scripts.ScriptRunner = p.scripts
        ic_light_script: scripts.Script = [
            script
            for script in script_runner.alwayson_scripts
            if script.title() == "IC Light"
        ][0]
        args = p.script_args[ic_light_script.args_from : ic_light_script.args_to]
        #assert len(args) == 1
        try:
            converted_dict = cls.convert_args_to_dict(args)
            #print(f"Converted dict: {converted_dict}")  # 打印转换后的字典
            return ICLightArgs(**converted_dict)
            #return ICLightArgs(**cls.convert_args_to_dict(args))
            #return ICLightArgs(**args[0])
            #input_fg=np.zeros(shape=[1, 1, 1], dtype=np.uint8),
        except Exception as e:
            print(f"{e}")  # 打印出详细的错误信息
        





# 多进程
    import multiprocessing

    def f(x):
        return x * x

    if __name__ == '__main__':
        with multiprocessing.Pool(5) as p:
            result = p.map(f, [1, 2, 3])
            print(result)

这将输出 [1, 4, 9]，表示对输入值 [1, 2, 3] 进行了并行处理。1







# 多线程下载处理

## 无效
    import multiprocessing

    def my_function(data):
        # 这是你的函数代码，假设它处理数据并返回结果
        result = data * data  # 示例操作
        return result

    def worker(data_queue, result_queue):
        while True:
            data = data_queue.get()
            if data is None:
                break
            result = my_function(data)
            result_queue.put(result)

    if __name__ == "__main__":
        # 创建队列来传递数据和结果
        data_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        # 创建进程池
        num_processes = 10
        processes = []
        for _ in range(num_processes):
            p = multiprocessing.Process(target=worker, args=(data_queue, result_queue))
            p.start()
            processes.append(p)

        # 将数据放入队列
        data_items = list(range(100))  # 示例数据
        for item in data_items:
            data_queue.put(item)

        # 发送终止信号
        for _ in range(num_processes):
            data_queue.put(None)

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print("处理结果:", results)


代码说明：
my_function(data): 这是你定义的需要多进程处理的函数。这里的示例将数据平方，但你可以用实际的处理逻辑替换它。

worker(data_queue, result_queue): 每个进程将执行这个函数。它从 data_queue 中取出数据，处理它，然后将结果放入 result_queue。通过将 None 放入队列作为结束信号，进程可以知道何时停止。

if __name__ == "__main__":: 这个检查确保代码在模块被导入时不会执行，只有在脚本直接运行时才执行。这个块中定义了进程池，数据队列和结果队列，创建并启动进程，放入数据，发送结束信号，等待进程完成，并收集结果。

注意事项：
数据队列 (data_queue): 用于传递待处理的数据。
结果队列 (result_queue): 用于收集每个进程处理后的结果。
终止信号 (None): 用于通知进程队列处理完毕，可以退出。

## pandas

在处理 Excel 文件（如使用 openpyxl 库）时，通常不建议使用多进程（multiprocessing）来加速 iter_rows 或类似迭代操作，因为 openpyxl 的读写操作并不是 CPU 密集型的，而是受限于 I/O（输入/输出）速度，且 Excel 文件通常不是线程或进程安全的。

不过，如果你的目的是处理 Excel 文件中的数据，并且处理逻辑本身可以并行化（即每个行的处理不依赖于其他行的结果），你可以考虑使用多线程（threading）而不是多进程，因为 Python 的全局解释器锁（GIL）在多线程环境下对 I/O 密集型任务的影响较小。但请注意，即使是多线程，由于 GIL 的存在，对于纯 CPU 密集型任务，Python 的多线程通常也不会带来显著的性能提升。

并且这些操作可以独立于 Excel 文件的读取进行，你可以考虑以下策略：

    使用多线程读取 Excel 数据：虽然这通常不会直接加速读取过程，但你可以并行地处理读取到的数据。
    分块处理：如果可能，尝试将 Excel 文件分块读取和处理，而不是一次性将所有数据加载到内存中。
    使用更高效的数据处理库：考虑使用如 pandas 这样的库来处理 Excel 数据，它通常比直接使用 openpyxl 进行迭代更高效。
    多进程处理（如果必须）：如果确实需要多进程，并且你的处理逻辑非常耗时且可以并行化，你可以考虑将 Excel 文件中的数据读取到内存中，然后将数据分块发送到不同的进程中处理。但请注意，这将需要更多的内存和复杂的同步机制。


这个示例中，我们实际上并没有加速 Excel 文件的读取过程，而是加速了读取后数据的处理过程。如果你的瓶颈在于 Excel 文件的读取，那么使用多线程或多进程可能不会带来显著的性能提升。


    import pandas as pd  
    from concurrent.futures import ThreadPoolExecutor  
    
    # 读取 Excel 文件  
    df = pd.read_excel('your_file.xlsx')  
    
    # 定义处理函数  
    def process_row(row):  
        # 这里是你的处理逻辑  
        # 假设只是简单地打印出来  
        print(row)  
    
    # 使用 ThreadPoolExecutor 来并行处理数据  
    with ThreadPoolExecutor(max_workers=4) as executor:  
        for _, row in df.iterrows():  
            executor.submit(process_row, row)



## 线程数

我们可以通过Python来查询CPU的线程数，这通常指的是CPU支持的并发线程的最大数量，这取决于CPU的核心数和每个核心支持的线程数（对于支持超线程的处理器）。

以下是一个使用Python查询CPU线程数的示例方法，它使用了os和multiprocessing模块：

使用os.cpu_count()
os.cpu_count()函数返回CPU的核心数，但它可能不会考虑到超线程技术。不过，在很多情况下，这个函数返回的值足够用来估计可以并发运行的线程数。

python
import os  
  
print(f"Number of CPU cores: {os.cpu_count()}")
使用multiprocessing.cpu_count()
与os.cpu_count()相似，multiprocessing.cpu_count()也返回CPU的核心数。但在某些情况下，它可能会给出更精确的关于可用于并发的处理器单元数量的信息。

python
import multiprocessing  
  
print(f"Number of CPU cores: {multiprocessing.cpu_count()}")



## 守护线程

使用thread开启多线程，使用ctrl c会无法完全关闭吗


在使用Python的threading模块开启多线程时，确实存在使用Ctrl+C（即SIGINT信号）无法完全关闭所有线程的情况。这是因为Python中，只有主线程默认能够捕获到SIGINT信号，而其他非守护（daemon）线程在默认情况下并不会响应这一信号。当主线程接收到SIGINT信号并退出时，非守护线程会继续执行，直到它们自己完成或者因为其他原因（如系统资源耗尽）被终止。

要解决这个问题，可以采取以下几种策略：

1.设置守护线程（Daemon Threads）
将线程设置为守护线程，这样当主线程退出时，所有守护线程也会被强制终止。但是，这种方法有一个缺点，即守护线程可能没有机会完成它们正在执行的任务或进行必要的清理工作。


    import threading  
    
    def my_function():  
        while True:  
            # 执行一些任务  
            pass  
    
    threads = []  
    for _ in range(5):  
        t = threading.Thread(target=my_function)  
        t.daemon = True  # 将线程设置为守护线程  
        t.start()  
        threads.append(t)  
    
    # 主线程继续执行其他任务或直接退出  
    # 当主线程退出时，所有守护线程也会随之退出

3.使用threading.Event控制线程退出
通过threading.Event对象，可以在主线程中设置一个标志，当接收到SIGINT信号时，修改这个标志，然后各个线程周期性地检查这个标志，如果发现需要退出，则执行清理工作并安全退出。


    import threading  
    import signal  
    import sys  
  
    def my_function(event):  
        while not event.is_set():  
            # 执行一些任务  
            pass  
        # 执行清理工作  
        print("线程退出")  
  
    def signal_handler(signal, frame):  
        print('You pressed Ctrl+C!')  
        event.set()  # 设置事件，通知所有线程退出  
    
    event = threading.Event()  
    
    threads = []  
    for _ in range(5):  
        t = threading.Thread(target=my_function, args=(event,))  
        t.start()  
        threads.append(t)  
    
    signal.signal(signal.SIGINT, signal_handler)  
    
    # 主线程可以等待所有线程完成，但在这个例子中，我们依赖于事件来控制退出  
    # for t in threads:  
    #     t.join()  
    
    # 注意：在实际应用中，如果主线程没有其他任务需要执行，  
    # 可以考虑使用while循环加上sleep来保持主线程运行，直到所有线程都通过事件退出。  
    # 但在这个简单的示例中，我们直接让主线程退出，因为事件处理函数会负责设置事件。



在使用多线程编程时，如果主线程（或某个关键线程）因为接收到Ctrl+C（SIGINT信号）而异常终止，可能会导致其他线程没有机会正常清理资源或执行其结束逻辑。这可能导致资源泄露、数据不一致或其他潜在问题。以下是一些处理和查看这些未正常退出线程的方法：






# parser










# 函数参数传入 地址式修改

p_num_list = [ [] for _ in data_chunks] 可以

但 p_num_list = [ 0 for _ in data_chunks] 不行    
后期会发现没修改 无法使用

    threads = []  
    results = [[] for _ in data_chunks]  # 每个线程都有自己的结果列表（但在这个例子中，我们实际上只需要一个） 
    httperror_list = [[] for _ in data_chunks]
    p_num_list = [ [] for _ in data_chunks]
    n_num_list = [ [] for _ in data_chunks]
    timeout = 3    
    logger.info(f"normal process: timeout {timeout}")
    logger2.info(f"normal process: timeout {timeout}")
    # 启动线程  
    for i, chunk in enumerate(data_chunks):  
        # 注意：这里我们实际上只需要一个共享的结果列表，但为了符合题目要求，我们仍然为每个线程创建了一个  
        # 在实际应用中，你可以只传递一个results列表，并在每个线程中直接使用它（但那样就不是“各自使用独立列表”了）  
        t = threading.Thread(target=process_file_pandas_threading, args=(chunk, results[i], httperror_list[i], p_num_list[i], n_num_list[i], fileFolderPath, excel_path, local_pickle_path, logger, timeout))  
        



# 当前文件/夹 位置

import os
current_dir = os.getcwd()
print("当前工作目录的路径：", current_dir)



import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print("当前文件所在目录的完整路径：", current_dir)






    import os  
    
    
    script_path = __file__  
    只是文件名 不是完整文件路径
    
    # 打印完整路径  
    print("完整路径:", script_path)  
    

目录

    # 获取目录路径  
    dir_path = os.path.dirname(script_path)  
    
    # 打印目录路径  
    print("目录路径:", dir_path)

# 遍历某个文件夹下所有文件
方法1：使用os模块
os模块提供了很多与操作系统交互的功能，包括遍历目录。但是，为了递归地遍历所有子目录，你通常需要自己编写递归函数。

    python
    import os  
    
    def traverse_dir(directory):  
        for root, dirs, files in os.walk(directory):  
            for file in files:  
                print(os.path.join(root, file))  
    
    # 假设你想遍历的文件夹路径是'/path/to/your/directory'  
    directory_path = '/path/to/your/directory'  
    traverse_dir(directory_path)
这里，os.walk(directory)会生成一个三元组(dirpath, dirnames, filenames)，其中dirpath是一个字符串，表示当前正在遍历的这个目录的路径，dirnames是一个列表，内容是该目录下的所有子目录的名字（不包括路径），filenames是一个列表，内容是该目录下的所有非目录文件的名字。

两种方法各有优缺点，但pathlib因其面向对象的设计和简洁的API，在Python 3.4及以后的版本中得到了广泛推荐。不过，对于熟悉os模块的老Python开发者来说，os模块仍然是一个强大的工具。










# 设置python执行到某一行停止
    import sys
    print("程序开始")
    # 在此处执行某些代码
    sys.exit()
    print("程序结束")

raise


    print("程序开始")
    # 在此处执行某些代码
    raise SystemExit
    print("程序结束")



Traceback (most recent call last):
  File "distribute_data.py", line 34, in <module>
    print(script_path)
NameError: name 'script_path' is not defined


# 换行 \

bash和py 都可以使用       

conda create -n wiseft python=3.8 -y && conda activate wiseft && \
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 scikit-learn open_clip_torch numpy wandb torch-geometric pyyaml psutil  wilds matplotlib  tqdm  boto3 && \
pip install torch_scatter 

类似此bash 正确解析

或者是 

    command && {
        command
    }      

子进程启动










# 结尾