import subprocess

if __name__ == "__main__":
    result = subprocess.run(
        'vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath',
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode == 0:
        installation_path = result.stdout.strip()
        x64_build_tool_path = f'{installation_path}/VC/Auxiliary/Build/vcvars64.bat'
        x86_build_tool_path = f'{installation_path}/VC/Auxiliary/Build/vcvars32.bat'

        try:
            with open(x64_build_tool_path) as file:
                print('VS2019 C++ x64 build tools are installed.')

            with open(x86_build_tool_path) as file:
                print('VS2019 C++ x86 build tools are installed.')
        except FileNotFoundError:
            print('VS2019 C++ build tools are not installed.')
    else:
        print('Unable to locate installation of Visual Studio 2019.')
