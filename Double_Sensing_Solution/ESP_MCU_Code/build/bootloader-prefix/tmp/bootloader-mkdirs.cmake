# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/ESP/v5.3/esp-idf/components/bootloader/subproject"
  "D:/ESP32_Project/MAT_Current_Source/build/bootloader"
  "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix"
  "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix/tmp"
  "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix/src/bootloader-stamp"
  "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix/src"
  "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/ESP32_Project/MAT_Current_Source/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
