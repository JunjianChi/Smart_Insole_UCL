#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/message_buffer.h"
#include "wifi_sender.h"
#include "adc_continuous.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "esp_task_wdt.h"
#include "sensor_config.h"
#include "dac_continuous.h"

bool adc_task_status = false;
bool wifi_task_status = false;

void app_main(void) {
    // close watchdog
    esp_task_wdt_deinit();

    // Initialize NVS must be done before any other functions from nvs_flash
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Initialize GPIO for MUX conntrol
    init_gpio();

    // // This is to check the integrity of the heap, recommended to be called after the system is initialized
    // heap_caps_check_integrity_all(true);

    // // Initialize WiFi
    wifi_init_sta();
    udp_connection_init();

    // continuous_dac_init();
    // xTaskCreate(dac_dma_write_task, "dac_dma_write_task", 4096, dac_task_handle(), 5, NULL);

    xTaskCreatePinnedToCore(read_adc, "read_adc_task", 1024 * 80, NULL, 2, get_adc_task_handle(), tskNO_AFFINITY);








    
    // xTaskCreate(client_task, "client_task", 1024 * 40, NULL, 3, get_wifi_task_handle());

    // while (1) {
    //     if (wifi_is_connected() && !adc_task_status) {
    //         xTaskCreate(read_adc, "read_adc_task", 1024 * 80, NULL, 2, get_adc_task_handle());
    //         adc_task_status = true;
    //         ESP_LOGI("app_main", "ADC task created");
    //     }
    //     if (!wifi_is_connected() && adc_task_status) {
    //         vTaskDelete(*get_adc_task_handle());
    //         adc_task_status = false;
    //         ESP_LOGI("app_main", "ADC task deleted");
    //     }
    //     if (wifi_is_connected() && !wifi_task_status) {
    //         xTaskCreate(client_task, "client_task", 1024 * 80, NULL, 3, get_wifi_task_handle());
    //         wifi_task_status = true;
    //         ESP_LOGI("app_main", "WiFi task created");
    //     }
    //     if (!wifi_is_connected() && wifi_task_status) {
    //         vTaskDelete(*get_wifi_task_handle());
    //         wifi_task_status = false;
    //         ESP_LOGI("app_main", "WiFi task deleted");
    //     }
    //     vTaskDelay(500 / portTICK_PERIOD_MS);
    // }
}
