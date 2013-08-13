import setuptools
import setuptools.extension

#setuptools.dist.Distribution(dict(setup_requires='Cython'))
## `setup_requires` is parsed and acted upon immediately; from here on out
## the yoursharedsetuppackage is installed and importable.
## Thanks to Martijn Pieters http://stackoverflow.com/a/12061891/351084

#from Cython.Distutils import build_ext

#setuptools.setup(name='vowpal_porpoise',
      #description='Lightweight vowpal wabbit wrapper',
      #version='0.2',
      #author='Austin Waters',
      #author_email='austin.waters@gmail.com',
      #packages=setuptools.find_packages(),
      #install_requires=[],
      #ext_modules=[setuptools.extension.Extension(
              #"vw_c",
              #["src/vw.pyx", "src/vw_c.cpp"],
              #language="c++",
              #include_dirs=["/mnt/vowpal_wabbit/"],
              ## library_dirs=['/mnt/vowpal_wabbit/vowpalwabbit'],
              #libraries=['allreduce', 'vw', 'boost_program_options-mt'],
              #)
          #],
      #cmdclass={'build_ext': build_ext}
#)



from setuptools import setup, find_packages

setup(
    name = 'vowpal_porpoise',
    version = '0.3',
    author = 'Joseph Reisinger',
    author_email = 'joeraii@gmail.com',
    description='Lightweight vowpal wabbit wrapper',
    license = 'BSD',
    keywords = 'machine learning regression vowpal_wabbit',
    url = 'https://github.com/josephreisinger/vowpal_porpoise',
    packages = find_packages(),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires = [    # dependencies
        'numpy',
        'scikit-learn',
    ],
    tests_require = [    # test dependencies
    ]
)
