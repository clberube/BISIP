# -*- mode: python -*-

block_cipher = None


a = Analysis(['scipy.linalg.cython_lapack', 'scipy.special._ufuncs_cxx', 'McMCspectrIP.py'],
             pathex=['/Users/Charles/Documents/Python Scripts/spectrIPmc'],
             binaries=None,
             datas=None,
             hiddenimports=['scipy.linalg.cython_blas'],
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
          name='scipy.linalg',
          debug=False,
          strip=None,
          upx=True,
          console=False )
app = BUNDLE(exe,
             name='scipy.linalg.app',
             icon=None,
             bundle_identifier=None)
