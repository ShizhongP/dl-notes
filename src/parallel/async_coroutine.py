import time
import asyncio
async def task():
    print("Task is started")
    # 模拟耗时操作 比如读取磁盘的数据
    await asyncio.sleep(1)
    return "Task is done"
    
async def task2():
    print("Task2 is started")
    await asyncio.sleep(1)
    return "Task2 is done"

if __name__ == "__main__":
    # 创建事件循环
    loop = asyncio.get_event_loop()
    # 创建任务,封装到future中
    tasks = [asyncio.ensure_future(task()), asyncio.ensure_future(task2())]
    # 将任务加入事件循环
    loop.run_until_complete(asyncio.wait(tasks))
    
    #获取时间的结果
    for t in tasks:
        print(t.result())
    # 关闭事件循环
    loop.close()
