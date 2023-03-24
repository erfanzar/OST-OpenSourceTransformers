import os

if __name__ == '__main__':
    # Set the paths to the necessary tools
    CMAKE_PATH = "C:\\Program Files\\CMake\\bin\\cmake.exe"
    PYTHON_PATH = "D:\\Python3.11\\python.exe"
    BUILD_PATH = os.path.abspath("build")

    # Download and extract the source code
    # Replace SOURCE_URL with the URL of the source code package
    # Replace SOURCE_ARCHIVE with the filename of the source code package
    SOURCE_URL = "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.95.zip"
    SOURCE_ARCHIVE = "v0.1.95.zip"
    os.system(f"curl -L {SOURCE_URL} -o {SOURCE_ARCHIVE}")
    os.system(f"tar -xzf {SOURCE_ARCHIVE}")

    # Create the build directory
    os.makedirs(BUILD_PATH, exist_ok=True)

    # Change to the build directory
    os.chdir(BUILD_PATH)

    # Configure the build
    os.system(f"{CMAKE_PATH} -DPYTHON_EXECUTABLE={PYTHON_PATH} ..")

    # Build the package in release mode
    os.system(f"{CMAKE_PATH} --build . --config Release")

    # Install the built package
    os.system(f"{PYTHON_PATH} setup.py install")

    print("Done.")
