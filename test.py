from itertools import islice
start_value = 138411971
with open('C:\\plot_data.json') as file:
	for line in file:
		for i, s in enumerate(islice(line, start_value-200, start_value + 200)):
			if i == 200:
				print()
			print(s, end="")
