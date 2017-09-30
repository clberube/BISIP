# -*- mode: python -*-

block_cipher = None


a = Analysis(['BISIP_GUI.py'],
             binaries=None,
             datas=None,
             hiddenimports=['scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
            			'tkinter',
                    		],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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
          upx=False,
          console=True, icon='casino.ico', version='versionfile')
