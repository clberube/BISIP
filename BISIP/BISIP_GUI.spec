# -*- mode: python -*-

block_cipher = None


a = Analysis(['BISIP_GUI.py'],
             pathex=['C:\\Users\\Charles\\Documents\\Python Scripts\\BISIP\\BISIP'],
             binaries=None,
             datas=None,
             hiddenimports=['scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack', 'scipy.special._ufuncs_cxx'],
             hookspath=None,
             runtime_hooks=None,
             excludes=None,
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
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
          strip=None,
          upx=True,
          console=True , icon='MC.ico')
