// #ifndef DAC_CONTINUOUS_H_
// #define DAC_CONTINUOUS_H_

// #include "sensor_config.h"
// #include <freertos/FreeRTOS.h>
// #include <freertos/task.h>
// #include <freertos/queue.h>

// void dac_dma_write_task(void *args);

// void continuous_dac_init(void);

// TaskHandle_t* dac_task_handle(void);

// #endif // DAC_CONTINUOUS_H_

#ifndef DAC_CONTINUOUS_H_
#define DAC_CONTINUOUS_H_

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "driver/dac_continuous.h"

// DAC配置相关常量
#define BUFFER_SIZE 2048
#define SAMPLE_RATE 100000  // DAC频率为200kHz
#define SINE_WAVE_FREQUENCY 100  // 正弦波频率为1kHz

// 函数声明
void generate_sine_wave(uint8_t *buffer, size_t buffer_size);
void dac_sine_wave_task(void *args);
void continuous_dac_init(void);

#endif // DAC_CONTINUOUS_H_