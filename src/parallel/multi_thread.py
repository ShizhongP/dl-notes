import threading 
import time
def run(id):
    print(f"Thread {id} is running")
    time.sleep(1)
    print(f"Thread {id} is done")

if __name__ == "__main__":
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=run, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
        
    print("All threads are done!")