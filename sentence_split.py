


def convert(lst):
    return ([i for item in lst for i in item.split()])
     
# Driver code

list213=[]
lst=['1']
def word_grouped(input_text):



			#lst =  ['Geeksforgeeks is a portal for geeks']
			lst[0]=input_text
			print(convert(lst))
			list213.clear()



			my_list = convert(lst)
			  
			# Yield successive n-sized
			# chunks from l.
			def divide_chunks(l, n):
			      
			    # looping till length l
			    for i in range(0, len(l), n): 
			        yield l[i:i + n]
			  
			# How many elements each
			# list should have
			n = 1
			  
			x = list(divide_chunks(my_list, n))
			print (x)


			#c=[['oh', 'yea', 'makes', 'sense'], ['Estas', 'enfermedad', 'un', 'cargo', 'poltico', 'tu', 'como', 'pblico', 'jesuischarlieytal'], ['old', 'men', 'finally', 'date', 'sarcasmsun', 'mar', 'ist'], ['sarinas', 'chanted', 'peacefully', 'deny', 'hypocrisysat', 'mar', 'ist']]

			for i12 in x:
			     print("for each list of the data",[' '.join(i12)])
			     l23=[' '.join(i12)]
			     list213.append(l23)
			return list213
			





	
