from distutils.core import setup

setup(
    name='meanshift-rs',
    version='0.3.0',
    packages=['py_meanshift'],
    package_dir={'py_meanshift': 'py_meanshift'},
    url='',
    license='MIT',
    author='Phillip Wenig',
    author_email='phillip.wenig@hpi.de',
    description='MeanShift Rust-Python binding'
)
