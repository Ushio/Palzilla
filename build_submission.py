import os
import shutil
import glob
from pathlib import Path

packagePath = Path('ushio_gpu')
packagePath.mkdir(parents=True, exist_ok=True)

for file in glob.glob("bin/*.dll"):
    shutil.copy2( file, str(packagePath / Path(file).name))

shutil.copy2( "bin/main_gpu.exe", str( packagePath / "main_gpu.exe" ) )
shutil.copytree("bin/kernels", str( packagePath / "kernels" ), dirs_exist_ok=True)

(packagePath / "assets").mkdir(parents=True, exist_ok=True)
for file in ["laminate_floor_02_diff_2k.jpg", "scene.abc"]:
    shutil.copy2( "bin/assets/" + file, str( packagePath / "assets" / file ) )

cuda_path = os.environ.get('CUDA_PATH')
print(cuda_path)
for nvrtc_dll in glob.glob(f"{cuda_path}/bin/nvrtc*.dll"):
    name = os.path.basename(nvrtc_dll)
    shutil.copy2( nvrtc_dll, str( packagePath / name ))

f = open(str(packagePath / "fps.txt"), 'w')
f.write('24')
f.close()

f = open(str(packagePath / "run.ps1"), 'w')
f.write('.\\main_gpu.exe')
f.close()