# Python 类的使用笔记

## 1. 类的创建

在 Python 中，类是面向对象编程的核心。你可以通过 `class` 关键字来定义一个类。类通常包含属性（类的变量）和方法（类的函数）。

### 类的基本结构

```python
class MyClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 注意当调用类自带的变量的时候需要self.
    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.") 
```

## 2. 创建类的对象

创建类的对象，也称为实例化类。你可以使用类名后跟括号来创建一个对象。

```python
person = MyClass("Alice", 30)
person.greet()  # 输出: Hello, my name is Alice and I am 30 years old.
```

## 3.类中的继承

类可以继承另一个类，继承意味着你可以在新类中复用父类的代码：
**继承父类的时候子类括号里面要写要继承父类的名称**

```python
class Student(MyClass):  # 继承 MyClass
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # 调用父类的构造函数
        self.student_id = student_id

    def study(self):
        print(f"{self.name} is studying.")


student = Student("Bob", 20, "S12345")
student.greet()  # 继承了父类的 greet 方法
student.study()  # 子类特有的 study 方法
```

## 4. 子类调用父类原有的方法和成员变量

1. 可以直接使用`父类.方法()`和`父类.成员变量`来调用
2. 或者使用`super().方法()`和`super().成员变量`来调用
