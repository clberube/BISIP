pyinstaller --hidden-import=scipy.linalg.cython_blas --hidden-import=scipy.linalg.cython_lapack --hidden-import=scipy.special._ufuncs_cxx --onefile BISIP_GUI.py

OR

pyinstaller bisip_gui.spec