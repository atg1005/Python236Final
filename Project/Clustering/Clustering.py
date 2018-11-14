import pandas as pd


if __name__ == '__main__':
    """Clustering is an alternative method to find the global max of the dataset."""
    # Attributes it wold make sense to know distance between (this data set does not have any)
    numericAttributeIndices = []
    #link given by data.world This is so I do not have to store data locally.
    data = pd.read_csv('https://query.data.world/s/a5hbsg7jsnkoumgrhou32l4vdgedj4')
    for i,row in data.iterrows():
        print(row)
        break
