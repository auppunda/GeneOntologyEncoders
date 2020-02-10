import sys
import csv

go_names = []
go_def = []

with open(sys.argv[1], 'r') as readfile:
	readcsv = csv.reader(readfile, delimiter='\t')
	for row in readcsv:
		go_names.append(row[0]+'\n')
		go_def.append(row[1]+'\n')


with open(sys.argv[2], 'w') as writefile:
	line = writefile.writelines(go_names[1:])


with open(sys.argv[3], 'w') as writefile:
	line = writefile.writelines(go_def[1:])


