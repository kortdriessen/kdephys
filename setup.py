from setuptools import setup

setup(
    name="kdephys",
    version="0.1",
    description="General tools for analyzing electrophysiology data",
    url="http://github.com/kortdriessen/kdephys",
    author="Kort Driessen",
    author_email="driessen2@wisc.edu",
    license="MIT",
    packages=["kdephys"],
    install_requires=[
        "pandas",
        "numpy",
        "xarray",
        "scipy",
        "tdt",
        "pyyaml",
        "matplotlib",
        "seaborn",
        "pandas_flavor",
        "spikeinterface",
        "polars",
        "lazy_loader",
        "netCDF4",
    ],
    zip_safe=False,
)
