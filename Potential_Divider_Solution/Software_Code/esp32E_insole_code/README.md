# Hardware Platform

ESP32-E

# Software Platform

idf 5.2.* +

# The Dependencies (develop on vscode)
You need to add by yourself in ./vscode/setting.json

```
// This part should add to the exist part, you need modlify your own {USERNAME}
{
    "idf.userPath": "/path/to/your/home/directory"
}

// Mac user
{
    "idf.userPath": "/Users/{USERNAME}"
}
// Windows user
{
    "idf.userPath": "C:\\Users\\{USERNAME}"
}

```

```
// This part should add to the exist part， direct paste
{
    "idf.espIdfPath": "${config:idf.userPath}/esp/esp-idf",
    "idf.pythonBinPath": "${config:idf.userPath}/.espressif/python_env/idf5.2_py3.11_env/bin/python",
    "idf.toolsPath": "${config:idf.userPath}/.espressif"
}

```

You also need to add by yourself in ./vscode/c_cpp_properties.json


```
{
    "configurations": [
        {
            "name": "ESP-IDF",
            "compilerPath": "${config:idf.toolsPath}/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc",
            "compileCommands": "${config:idf.buildPath}/compile_commands.json",
            "includePath": [
                "${config:idf.espIdfPath}/components/**",
                "${config:idf.espIdfPathWin}/components/**",
                "${workspaceFolder}/**"
            ],
            "browse": {
                "path": [
                    "${config:idf.espIdfPath}/components",
                    "${config:idf.espIdfPathWin}/components",
                    "${workspaceFolder}"
                ],
                "limitSymbolsToIncludedHeaders": true
            }
        }
    ],
    "version": 4
}
```