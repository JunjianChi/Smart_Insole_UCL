#ifndef _SENSOR_CONFIG_H_
#define _SENSOR_CONFIG_H_

// Include the sdkconfig header
#include "sdkconfig.h"
#include "esp_adc/adc_continuous.h"

#define WR01 CONFIG_WR01
#define WR02 CONFIG_WR02
#define A0   CONFIG_A0
#define A1   CONFIG_A1
#define A2   CONFIG_A2
#define A3   CONFIG_A3
#define A4   CONFIG_A4

#if CONFIG_MAT_SENSOR
    #define MAT_SENSOR
    #define TAG  "esp32_sensor_mat"
    // if not staic it should be defined in the main.c and extern in the header file
#endif

#if CONFIG_INSOLE_SENSOR
    #define INSOLE_SENSOR
    #define TAG  "esp32_sensor_insole"
#endif

#define ADC_CHANNEL    CONFIG_ADC_CHANNEL
#define SAMPLE_RATE    (200*1000)
// #define COLUMN_NUM     CONFIG_SENSOR_COL_SIZE
// #define ROW_NUM        CONFIG_SENSOR_ROW_SIZE
#define COLUMN_NUM     16
#define ROW_NUM        16
#define AVERAGE_NUM    CONFIG_ADC_AVERAGE_NUM
// #define AVERAGE_NUM    4
#define READ_LEN       (AVERAGE_NUM * SOC_ADC_DIGI_RESULT_BYTES)

#if CONFIG_DATA_ENCODE_ASCII
    #define DATA_ENCODE_ASCII
    #define ADC_DATA_SIZE_IN_ASCII 4
    // Type 1
    #if CONFIG_DATA_PACKAGE_COLUMN
        #define COLUMN_HEADER_SIZE 4
        #define PAYLOAD_SIZE (ROW_NUM * ADC_DATA_SIZE_IN_ASCII + COLUMN_HEADER_SIZE)
    #endif

    // Type 2
    #if CONFIG_DATA_PACKAGE_FRAME
        // #define PAYLOAD_SIZE (ROW_NUM * COLUMN_NUM * ADC_DATA_SIZE_IN_ASCII)
        #define PAYLOAD_SIZE (ROW_NUM * COLUMN_NUM * ADC_DATA_SIZE_IN_ASCII+1)
    #endif
#endif

#if CONFIG_DATA_ENCODE_RAW
    #define DATA_ENCODE_RAW

    // Type 3
    #if CONFIG_DATA_PACKAGE_COLUMN
        #define COLUMN_HEADER_SIZE 4
        #define PAYLOAD_SIZE (ROW_NUM * SOC_ADC_DIGI_RESULT_BYTES + COLUMN_HEADER_SIZE)
    #endif

    // Type 4
    #if CONFIG_DATA_PACKAGE_FRAME
        #define PAYLOAD_SIZE (ROW_NUM * COLUMN_NUM - 1)
    #endif

#endif

#endif  // _SENSOR_CONFIG_H_
