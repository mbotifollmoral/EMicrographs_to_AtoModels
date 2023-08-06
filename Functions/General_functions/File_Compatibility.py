# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:21:38 2023

@author: Marc
"""

'''
Functions for addressing file format changes when needed, typically between
cell file formats or probably micrograph formats
'''


from ase import data as asedata



directory = 1
filename = 1


def ACE_to_uce(
        directory, filename):
    '''
    Function changing the unit cell format exported from CaRine, and
    turning it into uce (unit cell Enzo) format which is the input for
    the dll crystal loader

    Parameters
    ----------
     : TYPE
        DESCRIPTION.

    Returns
    -------
    file_write, being the full name (directory included) of the new file
    generated, with extension .uce

    '''
        
    
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
                        # if line==file[0]:
                        #     coordinates0[i]=(float(words[i+2][:division])/float(words[i+2][division+1:]))
                        #     print('coordinates0[i]')
                        #     print(coordinates0[i])
                        # coordinates[i]=(float(words[i+2][:division])/float(words[i+2][division+1:]))-coordinates0[i]
                        coordinates[i]=(float(words[i+2][:division])/float(words[i+2][division+1:]))
                    else:
                        # if line==file[0]:
                        #     coordinates0[i]=float(words[i+2])
                        # coordinates[i]=float(words[i+2])-coordinates0[i] 
                        coordinates[i]=float(words[i+2])
                    
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
        
    def conversor(
            directory, filename):
        
        file_directory=directory+ '\\' + filename
        file_open=open(file_directory, encoding='ansi')
        file=(file_open.read()).splitlines()
        
        n_atoms='NATOM   '+str(len(search_atoms(file)))
        lengths_index=search_index(file, 'a=')
        lengths='CELL'+write_values(file[lengths_index])
        angles=write_values(file[lengths_index+1])
        symmetry_index=search_index(file, 'Group Number=')
        symmetry='RGNR'+write_values(file[symmetry_index])
        end_document=n_atoms+'\n'+lengths+angles+'\n'
        atoms=search_atoms(file)
        for i in atoms:
            end_document+=i+'\n'
        end_document+=symmetry+'\nSTN   0'
           
        file_write=directory+ '\\' +filename[:filename.find('.')]+'.uce'
        file=open(file_write,'w+')
        file.write(end_document)
        file.close()
        
        return file_write
    
    file_write = conversor(
        directory, filename)
    
    return file_write
    
    
