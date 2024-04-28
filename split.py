import json
from random import shuffle

def split_json(filename, test_size=0.2):
  """
  Splits a JSON file into training and test sets.

  Args:
      filename: Path to the JSON file.
      test_size: Proportion of data for the test set (default: 0.2).

  Returns:
      A dictionary containing training and test data as lists.
  """

  with open(filename, 'r') as f:
    data = json.load(f)

  shuffle(data)  # Shuffle data for random distribution

  split_point = int(len(data) * (1 - test_size))
  train_data = data[:split_point]
  test_data = data[split_point:]

  return {'train': train_data, 'test': test_data}

# Example usage
data = split_json('C:/Users/Beste/Desktop/AAB/dataset.json')

with open('C:/Users/Beste/Desktop/AAB/train_data.json', 'w') as f:
  json.dump(data['train'], f)

with open('C:/Users/Beste/Desktop/AAB/test_data.json', 'w') as f:
  json.dump(data['test'], f)

print("Data split completed! Training and test sets created.")