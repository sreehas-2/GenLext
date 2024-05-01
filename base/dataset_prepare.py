import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# making folders
outer_names = ['test','train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data',outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data',outer_name,inner_name), exist_ok=True)

# to keep count of each category
counts = {
    'angry': 0,
    'disgusted': 0,
    'fearful': 0,
    'happy': 0,
    'sad': 0,
    'surprised': 0,
    'neutral': 0,
    'angry_test': 0,
    'disgusted_test': 0,
    'fearful_test': 0,
    'happy_test': 0,
    'sad_test': 0,
    'surprised_test': 0,
    'neutral_test': 0
}

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        img.save(f'data/train/{inner_names[df["emotion"][i]]}/im{counts[inner_names[df["emotion"][i]]]+1}.png')
        counts[inner_names[df["emotion"][i]]] += 1

    # test
    else:
        img.save(f'data/test/{inner_names[df["emotion"][i]]}/im{counts[inner_names[df["emotion"][i]]+"_test"]+1}.png')
        counts[inner_names[df["emotion"][i]]+"_test"] += 1

print("Done!")
