#include "dac_continuous.h"
#include "driver/dac_oneshot.h"
#include "esp_log.h"

#define MAX_DAC_VOLTAGE 3300  // 最大电压 3.3V
#define DAC_RESOLUTION 255    // 8位 DAC，最大值 255

static const char *TAG = "DAC";

static dac_oneshot_handle_t dac1_handle = NULL;
static dac_oneshot_handle_t dac2_handle = NULL;

// 初始化 DAC One-Shot 模式
void init_dac() {
    dac_oneshot_config_t dac1_cfg = {
        .chan_id = DAC_CHANNEL_1, // GPIO 25
    };
    ESP_ERROR_CHECK(dac_oneshot_new_channel(&dac1_cfg, &dac1_handle));

    dac_oneshot_config_t dac2_cfg = {
        .chan_id = DAC_CHANNEL_2, // GPIO 26
    };
    ESP_ERROR_CHECK(dac_oneshot_new_channel(&dac2_cfg, &dac2_handle));

    ESP_LOGI(TAG, "DAC One-Shot mode initialized.");
}

// 设置 DAC 输出电压（0-3.3V）
void set_dac_voltage(dac_channel_t channel, uint16_t millivolts) {
    if (millivolts > MAX_DAC_VOLTAGE) {
        ESP_LOGW(TAG, "Voltage too high: %d mV, clamping to 3.3V", millivolts);
        millivolts = MAX_DAC_VOLTAGE; // 限制最大值
    }

    uint8_t dac_value = (millivolts * DAC_RESOLUTION) / MAX_DAC_VOLTAGE; // 转换到 0-255

    dac_oneshot_handle_t handle = (channel == DAC_CHANNEL_1) ? dac1_handle : dac2_handle;
    if (handle == NULL) {
        ESP_LOGE(TAG, "DAC channel %d not initialized!", channel);
        return;
    }

    ESP_ERROR_CHECK(dac_oneshot_output_voltage(handle, dac_value));

    // ESP_LOGI(TAG, "Set DAC%d to %d mV (DAC value: %d)", channel + 1, millivolts, dac_value);
}