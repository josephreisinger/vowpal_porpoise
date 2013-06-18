import setuptools
import setuptools.extension

setuptools.dist.Distribution(dict(setup_requires='Cython'))
# `setup_requires` is parsed and acted upon immediately; from here on out
# the yoursharedsetuppackage is installed and importable.
# Thanks to Martijn Pieters http://stackoverflow.com/a/12061891/351084

from Cython.Distutils import build_ext

setuptools.setup(name='vowpal_porpoise',
      description='Lightweight vowpal wabbit wrapper',
      version='0.2',
      author='Austin Waters',
      author_email='austin.waters@gmail.com',
      packages=setuptools.find_packages(),
      install_requires=[],
      ext_modules=[setuptools.extension.Extension(
              "vw_c",
              ["src/vw.pyx", "src/vw_c.cpp"],
              language="c++",
              include_dirs=["/mnt/vowpal_wabbit/"],
              # library_dirs=['/mnt/vowpal_wabbit/vowpalwabbit'],
              libraries=['allreduce', 'vw', 'boost_program_options-mt'],
              )
          ],
      cmdclass={'build_ext': build_ext}
)
