from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='llm_bench',
    version='0.4.32',
    description='LLM Benchmark',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/b-Snaas/ollama-benchmark',
    packages=find_packages(),
    package_data={'llm_benchmark': ['data/*.yml', 'data/img/*.jpg']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'click>=8.0.0',
        'typer = 0.11.0',
        'ollama = 0.1.8',
        'pyyaml = 6.0.1',
        'requests = 2.31.0',
        'psutil = 5.9.8',
        'GPUtil = 1.4.0',
        'lib-platform = 1.2.10',
        'setuptools = 69.1.0',
        'speedtest-cli = 2.1.3',
    ],
    entry_points={
        'console_scripts': [
            'llm_bench = llm_bench.main:app',
        ],
    },
    # This line enables editable installs
    # With 'pip install -e .' equivalent
    # to install your package in editable mode
    # so changes in your source code are immediately reflected
    # without needing to reinstall
    options={'bdist_wheel': {'universal': True}},
    setup_requires=['setuptools>=69.1.0', 'wheel'],
    editable=True
)