from setuptools import setup

setup(name='my_explainers_ilieva',
      version='0.0.1',
      description='Classifier explainers',
      url='',
      author='Tsvetelina Ilieva',
      author_email='tsvetelina.ilieva.cs@gmail.com',
      license='N/A',
      packages=['my_explainers'],
      install_requires=[
         'pandas==1.0.4', 
         'shap==0.35.0', 
         'numpy==1.18.5', 
         'plotly==4.9.0', 
         'scipy==1.4.1', 
         'statistics==1.0.3.5', 
         'sklearn==0.0', 
         'ipywidgets== 7.5.1'
      ],
      zip_safe=False)