import ast
dictionary = ast.literal_eval("{'a': 1e-10, 'b': 2}")
print (type(dictionary))
print (dictionary)