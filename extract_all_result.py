import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def main():

    # make result folder
    result_path = os.path.join(os.getcwd(), 'result')
    os.makedirs(result_path, exist_ok=True)
    
    # check the data in folder
    data_path = os.path.join(os.getcwd(), 'data')
    data = os.listdir(data_path)
    data.sort()
    
    print("Extract All Result of Data")

    for vid in tqdm(data):
        with open(f"data/{vid}", "rb") as f:
            data = pickle.load(f)

        '''
            data:
                vid: imdb id of video
                num_shot: total # shots in video
                probs: probabilities of corresponding idx is boundary
                    (confidence score)
                gts: ground-truth labels
        '''

        num_shot = data["num_shot"]
        probs = data["prob"]
        gts = data["gt"]
        idx = np.arange(0, num_shot, dtype=int)

        # print(f"#shots: {num_shot}")
        
        result_video_path = os.path.join(result_path, vid)
        os.makedirs(result_video_path, exist_ok=True)

        # set start and end point
        for i in range(0,num_shot,200):
            # start = int(args.start)
            start = i
            end = start + 200
            assert start >= 0 and start < num_shot
            assert end >= 0
            if end >= num_shot:
                end = num_shot

            # visualize
            plt.figure(figsize=(17, 3))
            plt.title(vid + str(i))
            plt.xticks(range(0,end-start+1, 5), range(start, end+1, 5), rotation=45)
            plt.yticks([-0.02, .5, 1.02], [0., .5, 1.])
            plt.xlim([0, end-start])
            plt.ylim([-0.02, 1.02])
            plt.grid(True)
            plt.tight_layout()

            plt.scatter(idx[start:end+1]-start, gts[start:end+1], color="green")
            plt.plot(idx[start:end+1]-start, probs[start:end+1], color="black")

            # plt.show()
            plt.savefig(f'{result_video_path}/{vid}_{str(i)}.jpg')
            plt.close()
    

if __name__ == "__main__":
    main()