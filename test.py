a=b=c=d=e=1
for a in range(1,7):
	for b in range(1,7):
		for c in range(1,7):
			for d in range(1,7):
				for e in range(1,7):
						if a==1 & b==1 & c==1 & d==1 & e==1:
							print("%d%d%d%d%d" % (a,b,c,d,e))