import time
import multiprocessing


def run(i):
    print(f"Process {i} is running")
    time.sleep(1)
    print(f"Process {i} is done")
    
if __name__ == "__main__":
    
    #创建进程组 
    for i in range(5):
        p = multiprocessing.Process(target=run, args=(i,))
        p.start()
    print("All processes are started")
    
    #等待所有进程结束
    for p in multiprocessing.active_children():
        p.join()
    
    print("All processes are done!")