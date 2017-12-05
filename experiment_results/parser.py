import os

rewards_data = []

directory = os.fsencode('.')
for curfile in os.listdir(directory):
    filename = os.fsdecode(curfile)
    if filename.startswith('rewards_details'):
        print('reading file:', filename)
        content = []
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
