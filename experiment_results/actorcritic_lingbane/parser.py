import os
import matplotlib.pyplot as plt

def parse_rewards(filename):
    print('reading file:', filename)
    content, rewards_data = [], []
    with open(filename) as f:
        content = f.readlines()
    for line in content:
        if 'Outcome:' in line:
            cleaned = line[line.index("Outcome:"):]
            cleaned = cleaned.replace('[', '')
            cleaned = cleaned.replace(']', '')
            cleaned = cleaned.replace(',', '')
            cleaned = cleaned.replace('Outcome:', '')
            cleaned = cleaned.replace('reward:', '')
            cleaned = cleaned.replace('score:', '')
            rewards_data.append(tuple([int(x) for x in cleaned.split()]))

    multiplier = filename.split()[-1].replace('.txt', '')
            
    return rewards_data, multiplier

def main():
    directory = os.fsencode('.')
    for curfile in os.listdir(directory):
        filename = os.fsdecode(curfile)
        if filename.startswith('rewards_details') and filename.endswith('.txt'):
            rewards_data, multiplier = parse_rewards(filename)
            episodes = range(len(rewards_data))
            scores = [score for episode, reward, score in rewards_data]
            
            plt.clf()
            plt.ylabel('Score')
            plt.xlabel('Episode')
            plt.title('EP_MULTIPLIER = ' + multiplier)
            plt.plot(episodes, scores)
            plt.savefig(filename + '.png')

if __name__ == "__main__":
    main()
