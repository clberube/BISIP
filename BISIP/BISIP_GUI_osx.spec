# -*- mode: python -*-

block_cipher = None


a = Analysis(['BISIP_GUI.py'],
             pathex=['/Users/Charles/Desktop/BISIP/BISIP'],
             binaries=None,
             datas=None,
             hiddenimports=['scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
				],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4', 'PyQt4.QtCore', 'PyQt4.QtGui'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='BISIP_GUI',
          debug=False,
          strip=False,
          upx=True,
          console=True )
