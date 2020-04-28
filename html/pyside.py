def integrate():
	good=1
	file1=open('test.txt','w')
	file1.write("test")
	file1.close()
	file1=open('results.txt','w')
	if(good==1):
		file1.write("Good Squat")
	if(good==0): 
		file1.write("Bad Squat")
	file1.close()
	return good