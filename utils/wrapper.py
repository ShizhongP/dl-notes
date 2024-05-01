
import os 
import time 

def NvidiaWrapper(func):
    def wrapper(func):
        os.system("sh nvidia-watch.sh log.log")
        func()
        os.system('kill -9 $(cat nvidia-smi.pid)')
    return wrapper(func)

@NvidiaWrapper
def test():
    time.sleep(5)
    
    
test()