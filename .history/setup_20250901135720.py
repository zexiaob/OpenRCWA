import os
import glob
import pathlib
import setuptools


# -----------------------------
# Utilities
# -----------------------------
ROOT = pathlib.Path(__file__).parent.resolve()

def read_readme() -> str:
	readme_path = ROOT / "README.md"
	if readme_path.exists():
		try:
			return readme_path.read_text(encoding="utf-8")
		except Exception:
			# Fallback without encoding if the environment is odd
			return readme_path.read_text()
	return "Python Implementation of Rigorous Coupled Wave Analysis (OpenRCWA)."


def compute_version(base: str = "1.0.5") -> str:
	"""Return a PEP 440 compliant version.

	If CI provides GITHUB_RUN_NUMBER, append a post-release tag.
	"""
	run = os.getenv("GITHUB_RUN_NUMBER")
	if run and run.isdigit():
		return f"{base}.post{run}"
	return base


# -----------------------------
# Package data (within the rcwa package)
# -----------------------------
# Use patterns instead of enumerating files at build time. Wheel builders
# honor these when include_package_data=True.
PACKAGE_DATA = {
	"rcwa": [
		"nkData/**",
		"test/**",
		"examples/**",
		"source/*.py",
	]
}


long_description = read_readme()

setuptools.setup(
	name="rcwa",
	version=compute_version(),
	author="Jordan Edmunds (original author)",
	author_email="jordan.e@berkeley.edu",
	maintainer="Zeyuan Jin",
	description="Python Implementation of Rigorous Coupled Wave Analysis",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/zexiaob/OpenRCWA",
	project_urls={
		"Source": "https://github.com/zexiaob/OpenRCWA",
		"Issues": "https://github.com/zexiaob/OpenRCWA/issues",
		"Documentation": "https://zexiaob.github.io/OpenRCWA/" if os.getenv("DOCS_URL") else "",
	},
	packages=setuptools.find_packages(
		exclude=(
			"tests",
			"docs",
		)
	),
	include_package_data=True,
	package_data=PACKAGE_DATA,
	classifiers=[
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Programming Language :: Python :: 3.13",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering",
		"Topic :: Scientific/Engineering :: Physics",
	],
	python_requires=">=3.8",
	install_requires=[
		"numpy>=1.20.0",
		"matplotlib>=2.0.0",
		"pandas>=0.24.0",
		"scipy>=1.2.2",
		"pyyaml>=5.0.0",
		"progressbar2",
		"autograd",
	],
	extras_require={
		"dev": [
			"pytest>=6.2.2",
			"pytest-cov",
		]
	},
	license="MIT",
	license_files=["LICENSE.txt"],
)
