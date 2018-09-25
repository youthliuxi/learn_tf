from scipy.special import comb,perm

def factorial(n):
	if n == 0 or n == 1:
		return 1
	else:
		return (n*factorial(n-1))
x = 0
for i in range(2,2*5+1):
	print(comb(10,i))
	a = comb(2*5,i)*6**(10-i)/6**10
	print("%d个及以上出现的概率：%.15f" % (i,a))
	x += a

print(x)
# for x in range(2,8):
# 	print("%d个人开始玩儿：" % x)
# 	for i in range(x+1,x*5+1):
# 		a = comb(5*x,i)/6**i
# 		print("%d个及以上出现的概率：%.15f" % (i,a))
