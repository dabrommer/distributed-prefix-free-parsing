## Build

```
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```--input``` Path to input data

```-w``` Window size used for the rabin karp fingerprint

```-p``` Modulo used to determine splits
