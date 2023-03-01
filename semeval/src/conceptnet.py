import requests

obj = requests.get('http://api.conceptnet.io/c/en/example').json()
print(obj.keys())
print(obj['@context'])