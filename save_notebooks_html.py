#!/usr/bin/env python3
#
# Convert all available notebooks into html and place them in subfolder html
#
# Author: Matteo Ravasi <mrava@statoil.no>
#
# Date:   04-05-2018

import glob
import os


def convert_notebooks():
    if not os.path.exists('html'):
        os.makedirs('html')
    
    # find files
    directories = []
    notebooks = []
    for root, dirs, files in os.walk('.'):
    	if len(root)>=3:
    		if root[3]!='.' or '.DS_Store' in root:
    			for file in files:
    				if file[-5:]=='ipynb' and not 'checkpoint' in file:
    					directories.append(root)
    					notebooks.append(file)
    print('directories', directories)
    print('notebooks', notebooks)
    
    for directory, notebook in zip(directories, notebooks):
    	if not os.path.exists(os.path.join('html',directory[2:])):
    		os.makedirs(os.path.join('html',directory[2:]))
    	
    	notebook_html=notebook[:-5]+'html'
    	print('Converting %s into html/%s' % (os.path.join(directory, notebook), \
    		os.path.join('html',directory[2:], notebook_html)))
    	
    	os.system('jupyter-nbconvert --to html '+os.path.join(directory, notebook))
    	os.rename(os.path.join(directory, notebook_html), \
    		os.path.join('html', directory[2:], notebook_html))


if __name__ == "__main__":
	convert_notebooks()
