import json

path = 'data/test.soloist_filled_3.json'

data = json.load(open(path, 'r'))

data = data[:500]

json.dump(data, open('output/test.soloist_filled_3.json', 'w'))