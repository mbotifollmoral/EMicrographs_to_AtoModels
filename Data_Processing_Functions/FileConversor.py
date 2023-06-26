from ase import data as asedata

directory=r'E:\\Arxius varis\\PhD\\2nd_year\\Code\\unit_cells\\'

filename='InAs_Hex_SG.ACE'

def search_index(file,word):
    i=-1
    index=-1
    
    while i<0:
        index+=1
        i=file[index].find(word)
        
    return index

def search_atoms(file):
    index=search_index(file,'Atom')+1
    file=file[index::]
    atoms=[]
    coordinates0=[0,0,0]
    coordinates=[0,0,0]
    
    for line in file:
        words=line.split()
        if words[-1]=='NonEq.':
            atoms.append(words[0]) 
            atoms[-1]+='   '+str(asedata.atomic_numbers[atoms[-1]])
            for i in range(3):
                division=words[i+2].find('/')
                if division>-1:
                    if line==file[0]:
                        coordinates0[i]=(float(words[i+2][:division])/float(words[i+2][division+1:]))
                    coordinates[i]=(float(words[i+2][:division])/float(words[i+2][division+1:]))-coordinates0[i]
                else:
                    if line==file[0]:
                        coordinates0[i]=float(words[i+2])
                    coordinates[i]=float(words[i+2])-coordinates0[i] 
                atoms[-1]+='   '+str(coordinates[i]) 
            atoms[-1]+='   '+words[-2]+'   0.075   0'
            
    return atoms

def write_values(line):
    string=''
    line_spaces=line.split()
    
    for word in line_spaces:
       index=word.find("=")
       if index>-1: 
            string+='   '+str(word[index+1::])   
            
    return string
    
def conversor(filename):
    file_directory=directory+filename
    file_open=open(file_directory, encoding='ansi')
    file=(file_open.read()).splitlines()
    
    n_atoms='NATOM   '+str(len(search_atoms(file)))
    lengths_index=search_index(file, 'a=')
    lengths='CELL'+write_values(file[lengths_index])
    angles=write_values(file[lengths_index+1])
    symmetry_index=search_index(file, 'Group Number=')
    print(write_values(file[symmetry_index]))
    symmetry='RGNR'+write_values(file[symmetry_index])
    end_document=n_atoms+'\n'+lengths+angles+'\n'
    atoms=search_atoms(file)
    for i in atoms:
        end_document+=i+'\n'
    end_document+=symmetry+'\nSTN   0'
       
    file_write=directory+filename[:filename.find('.')]+'.uce'
    file=open(file_write,'w+')
    file.write(end_document)
    file.close()

    
conversor(filename)