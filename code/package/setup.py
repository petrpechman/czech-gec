from setuptools import setup, find_namespace_packages

with open("README.md") as fr:
    long_description = fr.read()

setup(
    name='petr-retag',
    version='0.0.1',
    packages=find_namespace_packages(include=["src.*"]),
    author='Pechman Petr',
    description="TODO.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://todo.com',
    python_requires=">=3.9",
    package_data={
        'src.retag.morphodita': ["*.dict"],
        'src.retag.vocabularies' : ["*.tsv"],
    },
    include_package_data=True,
    entry_points="""
        [console_scripts]
        retag=src.retag.retag:main_cli
        m2_to_tsv=src.m2_to_tsv.main:main_cli
        create_errors=src.retag.create_errors:main_cli
        create_order=src.create_order.create_order:main_cli
    """,
    install_requires=[
        'setuptools',
    ],
)
