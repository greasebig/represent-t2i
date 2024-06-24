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


# 字典键值对


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







# 结尾