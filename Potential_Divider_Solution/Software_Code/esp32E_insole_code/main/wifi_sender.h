#ifndef WIFI_SENDER_H
#define WIFI_SENDER_H

#include <stdbool.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "lwip/sockets.h"

// Include the sdkconfig header
#include "sdkconfig.h"

// WiFi configurations from Kconfig
#define WIFI_SSID "dd-wrt"
#define WIFI_PASS "0000000000"
#define SERVICE_ADDRESS "192.168.137.1"
#define SERVICE_PORT 44444

// WiFi event group and connection status flag
extern EventGroupHandle_t wifi_event_group;
extern const int WIFI_CONNECTED_BIT;
extern volatile bool wifi_connected;

void wifi_init_sta(void);
int udp_connection_init(void);
bool wifi_is_connected(void);
bool udp_is_connected(void);
TaskHandle_t* get_wifi_task_handle(void);
int send_udp_sock(int sock_get, const void *data, size_t size);
int send_udp(const void *data, size_t size);

// 1. wifi_init_sta
// 2. udp_connection_init
// 3. wifi_is_connected
// 4. udp_is_connected

#endif // WIFI_SENDER_H
