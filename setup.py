try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    "sklearn",
    "scipy",
    "numpy",
    "matplotlib"
]

test_requirements = []

setup(
    name='pseudo-labeller',
    version='0.0.1',
    description="Pseudo labelling - Semisupervised Learning",
    url='https://github.com/shubhamjn1/Pseudo-Labelling---A-Semi-supervised-learning-technique',
    packages=[
        'frameworks'
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='pseudo-labeller',
)
