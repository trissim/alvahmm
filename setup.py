#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import setuptools
###
with open('README.md', 'r', encoding = 'utf-8') as file_reading:
    long_description_file = file_reading.read()
###
setuptools.setup(name = 'alva_machinery',
                 version = '0.0.6',
                 description = 'Home_made machinery for image and seq analysis',
                 author = 'Alvason Zhenhua Li',
                 author_email = 'alvasonli@gmail.com',
                 url = 'https://github.com/alvason',
                 classifiers = [
                                ### How mature is this project?
                                ### Status: '3 - Alpha', '4 - Beta', '5 - Production/Stable'
                                'Development Status :: 3 - Alpha',
                                'Intended Audience :: Science/Research',
                                'Intended Audience :: Developers',
                                'License :: OSI Approved :: MIT License',
                                'Programming Language :: Python :: 3',
                               ],
                  packages = setuptools.find_packages(),
                  python_requires = '>=3.6', 
                  ### Optional +++
                  project_urls = {
                                  'Bug Tracker for RRS': 'https://github.com/alvason/identifying_neurite_by_RRS/issues',
                            'Bug Tracker for Tcell': 'https://github.com/alvason/visualizing_Tcell_distribution/issues',
                                 },
                  long_description = long_description_file,  
                  long_description_content_type = 'text/markdown',
                  ### Optional ---
                 )


# In[ ]:




