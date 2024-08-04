import csv
import random

# Function to extract strings from the specified columns in the CSV file
def extract_columns(file_path):
    extracted_data = []

    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            quantized_path = row['quantized_path']
            performance_path = row['performance_path']
            extracted_data.append(f"{quantized_path}|{performance_path}")

    return extracted_data

# Function to split data into train, validation, and test sets
def split_data(data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)
    
    train_data = data[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    test_data = data[train_size + validation_size:]
    
    return train_data, validation_data, test_data

# Path to the input CSV file
input_file = './paired-dataset-5/paths/dtw-filter-5.csv'  # Change this to your actual file path

# Extract the data
extracted_data = extract_columns(input_file)

# Split the data
train_data, validation_data, test_data = split_data(extracted_data)

# Write the train, validation, and test data to separate text files
with open('./paired-dataset-5/paths/train.txt', 'w') as file:
    for entry in train_data:
        file.write(f"{entry}\n")

with open('./paired-dataset-5/paths/validation.txt', 'w') as file:
    for entry in validation_data:
        file.write(f"{entry}\n")

with open('./paired-dataset-5/paths/test.txt', 'w') as file:
    for entry in test_data:
        file.write(f"{entry}\n")

print("Train data saved to train.txt")
print("Validation data saved to validation.txt")
print("Test data saved to test.txt")
