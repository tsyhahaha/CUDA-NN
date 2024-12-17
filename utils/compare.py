import os
import numpy as np

num=500

def load_data(file_path):
    return np.loadtxt(file_path)

def calculate_accuracy(folder_a, folder_b):
    correct_count = 0
    total_count = 0
    
    for i in range(num):
        a_file_path = os.path.join(folder_a, f'{i}.txt')
        b_file_path = os.path.join(folder_b, f'{i}.label.txt')
        
        if os.path.exists(a_file_path) and os.path.exists(b_file_path):
            predictions = load_data(a_file_path)
            labels = load_data(b_file_path)

            correct_count += np.sum(predictions == labels)
            total_count += len(labels)

    if total_count == 0:
        return 0.0

    accuracy = correct_count / total_count
    return accuracy

if __name__ == '__main__':
    folder_a = '/home/tsyhahaha/CUDA-NN/data/cuout'
    folder_b = '/home/tsyhahaha/CUDA-NN/data/beat'
    
    accuracy = calculate_accuracy(folder_a, folder_b)
    print(f'Accuracy: {accuracy * 100:.2f}%')
