# Insole_mocap

This project focuses on capturing data from an insole equipped with 255 pressure sensors. The data is read using two 32x32 multiplexers, mapped into a grid format, and combined with depth camera data for machine learning (ML) applications, particularly for gait analysis.

## Insole_ver1_demo

<img src="./mdfile//Insole_movie.gif" width="600" height="350"/><br/>

## Table of Contents

1. [Project Structure](#project-structure)
2. [Hardware Setup](#hardware-setup)
3. [Software Setup](#software-setup)
4. [Data Collection](#data-collection)
5. [Data Processing](#data-processing)
6. [Gait Analysis](#gait-analysis)
7. [Future Work](#future-work)

## Project Structure

The project is divided into the following components:

- **`hardware/`**: Contains files related to hardware setup, including connection diagrams and setup instructions.
- **`software/`**: Contains software dependencies and scripts for data collection and processing.
- **`data/`**: Stores raw and processed data files.
- **`docs/`**: Documentation related to the project.

## Hardware Setup

1. **Insole with Pressure Sensors**: The insole is equipped with 255 pressure sensors arranged in a grid format.
2. **MCU (Microcontroller Unit)**: The MCU is responsible for reading data from the insole. For MCU code, see the `MCU_code` folder.
3. **Depth Camera**: Captures depth data that is synchronized with the insole sensor readings.

### Connection Diagram

Continued...

### Depth Camera Setup

Continued...

## Software Setup

Continued...

## Data Collection

The `server.py` script handles data collection from the MCU. It sends the collected data to the PC via UDP. Ensure that the MCU and PC are properly connected to facilitate data transfer.

## Data Processing

Data collected from the sensors and depth camera is processed to create a comprehensive dataset for analysis. This involves:

- **Cleaning**: Removing noise and irrelevant data.
- **Normalization**: Standardizing the data format for consistent analysis.
- **Integration**: Combining sensor data with depth camera information.

## Gait Analysis

Machine learning techniques are applied to the processed data to analyze gait patterns. The analysis aims to identify various gait parameters, such as step length, stride frequency, and pressure distribution.

## Future Work

- **MCU Development**: Current testing is based on a 25x10 channel setup. Future versions will use a customized pattern for improved imaging and data accuracy.
- **Insole Design**: Planned improvements include the development of 3D-printed models for the insole and the addition of fabric between the flex PCB sensors to enhance durability and comfort.
- **Project Timeline**: 
  - **Completed**: Project structure and initial hardware setup.
  - **In Progress**: Software setup, data collection, and preliminary data processing.
  - **Upcoming**: Advanced data processing techniques, gait analysis enhancements, and further hardware development.





For detailed updates and progress, check the [Issues](https://github.com/JunjianChi/Insole_mocap/issues) and [Projects](https://github.com/JunjianChi/Insole_mocap/projects) sections of this repository.



