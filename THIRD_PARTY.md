# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed
from PyPI wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence).
No model weights or datasets are bundled: specparam fits every spectrum from scratch, and the 10-05
electrode template positions ship inside MNE-Python.

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| fooof (specparam) | 1.1.1 | Thomas Donoghue and the Voytek Lab | Apache-2.0 | https://github.com/fooof-tools/fooof |
| MNE-Python | 1.13.2 | Alexandre Gramfort and the MNE-Python contributors | BSD-3-Clause | https://github.com/mne-tools/mne-python |
| NumPy | 2.5.4 | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | SciPy Developers | BSD-3-Clause | https://scipy.org |
| Matplotlib | 3.11.2 | Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| contourpy | 1.4.0 | ContourPy Developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project | BSD-3-Clause | https://github.com/matplotlib/cycler |
| decorator | 5.3.1 | Michele Simionato | BSD-2-Clause | https://github.com/micheles/decorator |
| fonttools | 4.68.0 | Just van Rossum and fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| Jinja2 | 3.1.6 | Pallets | BSD-3-Clause | https://github.com/pallets/jinja |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| lazy-loader | 0.6 | Scientific Python | BSD-3-Clause | https://github.com/scientific-python/lazy-loader |
| MarkupSafe | 3.0.4 | Pallets | BSD-3-Clause | https://github.com/pallets/markupsafe |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Jeffrey A. Clark and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| platformdirs | 4.14.5 | platformdirs contributors | MIT | https://github.com/tox-dev/platformdirs |
| pooch | 1.9.0 | The Pooch Developers | BSD-3-Clause | https://github.com/fatiando/pooch |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |
| python-dateutil | 4.0.0 | dateutil contributors | BSD-3-Clause | https://github.com/dateutil/dateutil |
| requests | 2.34.2 | Kenneth Reitz and contributors | Apache-2.0 | https://github.com/psf/requests |
| certifi | 2026.10.23 | Kenneth Reitz | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | Ahmed Tahri | MIT | https://github.com/jawah/charset_normalizer |
| idna | 3.20 | Kim Davies | BSD-3-Clause | https://github.com/kjd/idna |
| urllib3 | 2.8.0 | Andrey Petrov and contributors | MIT | https://github.com/urllib3/urllib3 |
| six | 1.17.0 | Benjamin Peterson | MIT | https://github.com/benjaminp/six |
| tqdm | 4.70.1 | tqdm developers | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |

Test-only (not in the image): pytest (MIT), edfio 0.4.18 (Apache-2.0, writes the synthetic fixture's EDF files).

The methods implemented in this app's own code (posterior dominant rhythm after Lodder & van Putten
2011; revised Brain Symmetry Index after van Putten 2007; band power and slowing ratios) were written
from the papers; no code was copied from their authors.
