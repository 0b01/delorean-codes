from hashlib import md5
from string import ascii_lowercase, digits
alphanumeric = ascii_lowercase + digits + "_"

target = "07ec795f46781fa1ece1fe3456fcd41e"
captcha = "w6dg"
github = "rickyhan"


def inp(t):
	m = md5()
	m.update(t)
	return m.hexdigest()

# for i in alphanumeric:
# 	totest = i + github + captcha
# 	if inp(totest) == target:
# 		print i
print target[:9]

i = str(int(target[:8], 16))
print i
# print str(i).decode('hex')