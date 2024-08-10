from setuptools import setup, find_packages

setup(
    name='High-Entropy-Alloy-Hardness',
    version='0.1',
    description='A machine learning project for predicting the hardness of high-entropy alloys No-Mb-Ta-W.',
    author='Manav Gupta',
    author_email='mm21b038@smail.iitm.ac.in',
    url='https://github.com/MM21B038/High-Entropy-Alloy-Hardness-Prediction',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
    'pandas==2.1.4',
    'numpy==1.26.4',
    'matplotlib==3.7.2',
    'seaborn==0.12.2',
    'scikit-learn==1.3.0',
    'tensorflow==2.14.0',
    'xgboost==1.7.4',
    'openpyxl==3.1.2'
],

    extras_require={
        'dev': [
            'jupyter',
            'ipython',
            'pytest',
            'flake8'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)