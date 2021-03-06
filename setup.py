from setuptools import setup, find_packages
import os


install_requires = [
    "pafprocess_ext",
    "torch",
    "torchvision",
    "pandas",
    "opencv-python",
    "numpy",
    "matplotlib",
    "scipy",
    "av"
]

dependency_links = [
    "".join(os.path.join(os.getcwd(), 'src', 'paf','pafprocess', 'dist', 'pafprocess_ext-1.0-py3.8-linux-x86_64.egg'))
]

setup(name='ai_coach',
      version='0.1',
      description='Coaching based on ai',
      author='Adam Olsson',
      #author_email='',
      install_requires=install_requires,
      dependency_links=dependency_links,
      packages=find_packages('src'),
      package_dir={'': 'src'}
     )

