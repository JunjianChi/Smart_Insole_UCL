// #include <math.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "driver/dac_continuous.h"
// #include "esp_log.h"
// #include "esp_check.h"
// #include <string.h>  

// // Define the constant value for DAC output
// #define CONSTANT_VOLTAGE_VALUE 128 
// #define EXAMPLE_ARRAY_LEN 2048 // Example buffer length

// static const char *TAG = "DAC Continuous";  // Define TAG for logging

// dac_continuous_handle_t handle = NULL;
// static TaskHandle_t s_task_handle = NULL;

// TaskHandle_t* dac_task_handle(void) {
//     return &s_task_handle;
// }

// void dac_dma_write_task(void *args)
// {  
//     // Get task handle
//     s_task_handle = xTaskGetCurrentTaskHandle();

//     size_t buf_len = EXAMPLE_ARRAY_LEN;
//     uint8_t *buffer = (uint8_t *)malloc(buf_len); // Allocate buffer memory

//     if (buffer == NULL) {
//         ESP_LOGE(TAG, "Failed to allocate memory for DAC buffer");
//         vTaskDelete(NULL);
//     }

//     // Fill the buffer with the constant voltage value
//     memset(buffer, CONSTANT_VOLTAGE_VALUE, buf_len);

//     while (1) {
//         // Check if handle is valid
//         if (handle == NULL) {
//             ESP_LOGE(TAG, "DAC handle is NULL");
//             free(buffer);
//             vTaskDelete(NULL);
//         }

//         // Write cyclically to DAC
//         ESP_ERROR_CHECK(dac_continuous_write_cyclically(handle, buffer, buf_len, NULL));

//         // You may adjust this delay or remove it if you want continuous output
//         vTaskDelay(pdMS_TO_TICKS(1000));
//     }

//     // Free buffer before task delete
//     free(buffer); 
//     vTaskDelete(NULL);  // Task should be deleted properly when done
// }

// void continuous_dac_init(void)
// {
//     dac_continuous_handle_t cont_handle;
//     dac_continuous_config_t cont_cfg = {
//         .chan_mask = DAC_CHANNEL_MASK_ALL,
//         .desc_num = 8,
//         .buf_size = 2048,
//         .freq_hz = 2000000, 
//         .offset = 0,
//         .clk_src = DAC_DIGI_CLK_SRC_DEFAULT,    
//         .chan_mode = DAC_CHANNEL_MODE_SIMUL,
//     };

//     // Allocate continuous channel
//     ESP_ERROR_CHECK(dac_continuous_new_channels(&cont_cfg, &cont_handle));

//     // Assign to global handle so task can use it
//     handle = cont_handle;

//     // Enable the channels in the group
//     ESP_ERROR_CHECK(dac_continuous_enable(cont_handle));

//     // Create DAC write task (pass NULL if no args are needed)
//     xTaskCreate(dac_dma_write_task, "dac_dma_write_task", 4096, NULL, 5, NULL);
// }

#include "dac_continuous.h"
#include "esp_log.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "DAC_CONTINUOUS";

void generate_sine_wave(uint8_t *buffer, size_t buffer_size)
{
    const double PI = 3.141592653589793;
    const double amplitude = 128; // DAC输出电压幅度范围（0-255）
    const double offset = 128; // DC偏置使正弦波的中心值在0-255范围内

    // 计算每个样本的角度步长
    double angle_step = 2 * PI * SINE_WAVE_FREQUENCY / SAMPLE_RATE;

    for (size_t i = 0; i < buffer_size; ++i) {
        double angle = i * angle_step;
        buffer[i] = (uint8_t)(amplitude * (sin(angle)) + offset);
    }
}

void dac_sine_wave_task(void *args)
{
    dac_continuous_handle_t handle = (dac_continuous_handle_t)args;

    uint8_t *buffer = (uint8_t *)malloc(BUFFER_SIZE);
    if (buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for DAC buffer");
        vTaskDelete(NULL);
    }

    // 生成正弦波数据
    generate_sine_wave(buffer, BUFFER_SIZE);

    ESP_ERROR_CHECK(dac_continuous_write_cyclically(handle, buffer, BUFFER_SIZE, NULL));

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    free(buffer);
    vTaskDelete(NULL);
}

void continuous_dac_init(void)
{
    dac_continuous_handle_t cont_handle;
    dac_continuous_config_t cont_cfg = {
        .chan_mask = DAC_CHANNEL_MASK_ALL,
        .desc_num = 8,
        .buf_size = BUFFER_SIZE,
        .freq_hz = SAMPLE_RATE,  // 设置DAC频率为200kHz
        .offset = 0,
        .clk_src = DAC_DIGI_CLK_SRC_DEFAULT,
        .chan_mode = DAC_CHANNEL_MODE_SIMUL,
    };

    // 初始化DAC连续输出
    ESP_ERROR_CHECK(dac_continuous_new_channels(&cont_cfg, &cont_handle));
    ESP_ERROR_CHECK(dac_continuous_enable(cont_handle));

    xTaskCreate(dac_sine_wave_task, "dac_sine_wave_task", 4096, cont_handle, 5, NULL);
}
