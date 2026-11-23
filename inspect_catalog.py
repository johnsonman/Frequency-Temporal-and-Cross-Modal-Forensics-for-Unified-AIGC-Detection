import urllib.request, json

url = 'https://www.thiings.co/api/catalog'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
r = urllib.request.urlopen(req)
data = json.load(r)

print('Top-level keys:', list(data.keys()))
print('Version:', data.get('version'))
print('Categories:', data.get('categories'))

items = data['items']
print('\nItems count:', len(items))
print('Type of items[0]:', type(items[0]).__name__)
print('items[0]:', items[0])
print('items[1]:', items[1])
print('items[2]:', items[2])
print('items[-1]:', items[-1])
