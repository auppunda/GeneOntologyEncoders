import sys
import json
import numpy as np
import csv

with open(sys.argv[1], "r") as readfile:
	vectors = readfile.readlines()


fout = open(sys.argv[3], "w")
fout.write("name\tgo_vectors\n")
with open(sys.argv[2], "r") as readfile:
	lines = readfile.readlines()
	vec_count = 0
	for line in lines:
		data = json.loads(line)
		vector = np.zeros(768)
		i = 0
		for feature in data['features']:
			values = feature['layers'][0]['values']
			num_values = np.array(values)
			vector += num_values
			i+=1
		vector/=i
		string_vec = ""
		for num in vector:
			string_vec += str(num) + " "
		string_vec = string_vec[:len(string_vec)-1] + "\n"
		entry = vectors[vec_count].strip('\n') + "\t" + string_vec
		vec_count+=1
		fout.write(entry)

fout.close()