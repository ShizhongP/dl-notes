import  pandas as pd
import matplotlib.pyplot as plt
import argparse

def draw(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.show()
    
    
def main():
    df = pd.read_csv(args.data_path,)
    data = df[' memory.used [MiB]']
    draw(data, " memory.used [MiB]")
    
if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./log.log")
    args = parser.parse_args()
    main()
    