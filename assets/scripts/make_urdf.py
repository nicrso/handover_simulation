from msilib.schema import Directory
import os

dir = './'

with open('template.urdf') as f: 
    lines = f.readlines()

 
for filename in os.listdir(dir):
    if filename.endswith(".obj"):
        new_filename = filename[:-3] + 'urdf'
        
        #create urdf file 
        outf = open(new_filename, 'w')
        new_lines = []
        for line in lines:
            line = line.replace('template.obj', filename)
            new_lines.append(line)
        outf.writelines(new_lines)
        outf.close()  
    