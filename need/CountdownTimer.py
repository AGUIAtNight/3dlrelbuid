import time


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # 执行被装饰的函数
        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间：{execution_time}秒")

        return result

    return wrapper


def measure_average_execution_time(n=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            total_execution_time = 0

            for i in range(n):
                start_time = time.time()

                # 执行被装饰的函数
                result = func(*args, **kwargs)

                end_time = time.time()
                execution_time = end_time - start_time
                total_execution_time += execution_time

            average_execution_time = total_execution_time / n
            print(f"平均执行时间：{average_execution_time}秒")

            return result

        return wrapper

    return decorator


# @measure_execution_time
# def my_code():
#     # 执行您的代码
#     # ...

# @measure_average_execution_time(n=10)
# def my_code_with_average_measurement():
#     # 执行您的代码
#     # ...
