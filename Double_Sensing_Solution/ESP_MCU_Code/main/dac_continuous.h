#ifndef DAC_CONTINUOUS_H
#define DAC_CONTINUOUS_H

#include <stdint.h>
#include "driver/dac_oneshot.h"

#define DAC1_CHANNEL DAC_CHANNEL_1  // GPIO 25
#define DAC2_CHANNEL DAC_CHANNEL_2  // GPIO 26

void init_dac(); // 初始化 DAC
void set_dac_voltage(dac_channel_t channel, uint16_t millivolts); // 设置 DAC 电压

#endif
