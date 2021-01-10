Data format
===========

It is important to respect this format when preparing your data files. We use
the format output of the SIP-Fuchs-III instrument and software.

.. code-block:: text

    Frequency, Amplitude,   Phase shift,  Amplit error, Phase error
    6.000e+03, 1.17152e+05, -2.36226e+02, 1.171527e+01, 9.948376e-02
    3.000e+03, 1.22177e+05, -1.46221e+02, 1.392825e+01, 1.134464e-01
    1.500e+03, 1.25553e+05, -9.51099e+01, 2.762214e+01, 2.199114e-01
    ........., ..........., ............, ............, ............
    ........., ..........., ............, ............, ............
    ........., ..........., ............, ............, ............
    4.575e-02, 1.66153e+05, -1.21143e+01, 1.947314e+02, 1.171115e+00
    2.288e-02, 1.67988e+05, -9.36718e+00, 3.306003e+02, 1.9669860+00
    1.144e-02, 1.70107e+05, -7.25533e+00, 5.630541e+02, 3.310889e+00

.. note::
    - Save your data in .csv, .txt, .dat or any other format. The extension is not important as long as it is a ASCII file.
    - Comma separation between columns is mandatory.
    - The order of the columns is very important (Frequency, Amplitude, Phase shift, Amplitude error, Phase error).
    - Phase units may be milliradians, radians or degrees.
    - Units are specified as an initialization argument (e.g. ph_units='mrad').
    - Amplitude units may be Ohm-m or Ohm, the data will be normalized.
    - A number of header lines may be skipped function argument (e.g. in this case headers=1).
    - Scientific or standard notation is OK.

Example data files
------------------

Several real-life data files are included with the BISIP package. These files contain
the complex resistivity spectra of mineralized rock samples.
Once you have installed BISIP in your Python environment, you may get the paths
to these data files using the :class:`DataFiles` class.

.. autoclass:: bisip.data.DataFiles
    :members:
    :show-inheritance:

The following example shows how to get the absolute file paths to the BISIP
data files in your Python installation.

.. code:: python

    from bisip import DataFiles

    files = DataFiles()  # initialize the dictionary
    print(files.keys())  # see the various data file names

    filepath = files['SIP-K389172']  # extract one path
    print(filepath)
