import os
# assign directory
directory = 'C:\\Users\\MLUser\\Downloads\\wav_chunks\\'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    
        print(filename)
