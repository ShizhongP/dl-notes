import time
import asyncio


async def task():
    print("Task is started")

    for i in range(0, 10):
        await asyncio.sleep(0.5)
        print(f"loop1: {i}")
    # 模拟耗时操作 比如读取磁盘的数据
    return "Task is done"


async def task2():
    print("Task2 is started")
    for i in range(0, 10):
        await asyncio.sleep(0.5)
        print(f"loop2: {i}")
    return "Task2 is done"


async def main1():
    res = await asyncio.gather(task(), task2(), task(), task2())
    print(res)



def main2():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [
        asyncio.ensure_future(task()),
        asyncio.ensure_future(task2()),
        asyncio.ensure_future(task()),
        asyncio.ensure_future(task2()),
    ]

    loop.run_until_complete(asyncio.wait(tasks))

    res = [t.result() for t in tasks]
    print(res)


if __name__ == "__main__":

    start = time.time()
    asyncio.run(main1())
    end = time.time()
    print("==================")
    print(f"main1 takes: {end - start} seconds")
    start = time.time()
    main2()
    end = time.time()
    print("==================")
    print(f"main2 takes: {end - start} seconds")
