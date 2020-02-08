import pickle

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


# example
if __name__ == '__main__':
    import numpy as np
    file_path = "dataset/negative/ffid_3.pickle"
    data = read_pickle(file_path)
    print("shape = ", np.shape(data))



