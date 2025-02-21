from setuptools import setup

setup(
    name="paten",
    version="0.5.9",
    packages=["paten"],
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "econml",
        "tableone",
        "pandas-gbq",
        "gspread",
        "tqdm",
        "matplotlib",
        "lightgbm"
    ]
)