 
import json
with open('bones_training.json', 'r') as f:
    data = json.load(f)
print(f'Loaded {len(data)} training examples')
print('First entry:')
print(json.dumps(data[0], indent=2))
