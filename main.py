import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", required=True)
    parser.add_argument("--start", required=True)
    args = parser.parse_args()

    vid = args.vid
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

    print(f"#shots: {num_shot}")
    
    # set start and end point
    start = int(args.start)
    end = start + 200
    assert start >= 0 and start < num_shot
    assert end >= 0 and end < num_shot

    # visualize
    plt.figure(figsize=(17, 3))
    plt.title(vid)
    plt.xticks(range(0,end-start+1, 5), range(start, end+1, 5), rotation=45)
    plt.yticks([-0.02, .5, 1.02], [0., .5, 1.])
    plt.xlim([0, end-start])
    plt.ylim([-0.02, 1.02])
    plt.grid(True)
    plt.tight_layout()

    plt.scatter(idx[start:end+1]-start, gts[start:end+1], color="green")
    plt.plot(idx[start:end+1]-start, probs[start:end+1], color="black")

    plt.show()
    

if __name__ == "__main__":
    main()