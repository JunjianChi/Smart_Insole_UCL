"""
──────────────────────────────────────────────────────────────────────────────
Real-time Crosstalk-free Rsensor Heat-map
──────────────────────────────────────────────────────────────────────────────
* Listens on UDP 192.168.137.1:8999 for alternating data frames
  ├─ Frame-0 (client == 0) : IA gain 1×, V+ = 1.60 V  →  used to compute the
                           true sensor resistance matrix (Rsensor, Ω)
  └─ Frame-1 (client == 1) : raw high-gain data         →  displayed for reference

* Pipeline
  1.  Receive 507 × 16-bit ADC words per frame (last word = client id)
  2.  Re-order the 253 valid channels with **lookup_table**
  3.  Map the 1-D list into a 33 × 15 geometry using **image_pattern**
  4.  Frame-0 → `rsensor_from_adc()`                      # crosstalk-free R (Ω)
──────────────────────────────────────────────────────────────────────────────
"""

import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import math

# UDP Configuration
UDP_IP = "192.168.137.1"  # IP address to bind to
UDP_PORT = 8999  # Port to listen on

BUFFER_SIZE = 508 * 2

MATRIX_ROW = 33  # Matrix size
MATRIX_COLUMN = 15

image_pattern = [[0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                ]
image_pattern = np.array(image_pattern)

lookup_table = [16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
                4,24,33,50,60,71,82,92,102,112,122,131,132,141,148,155,156,163,169,175,186,193,194,201,208,215,222,229,230,237,243,244,
                10,17,25,42,51,61,72,83,93,103,113,123,124,133,142,149,150,157,164,170,181,187,188,195,202,209,216,223,224,231,238,250,
                5,11,18,34,43,52,62,73,84,94,104,114,115,125,134,143,144,151,158,165,176,182,183,189,196,203,210,217,218,225,232,251,
                0,6,12,26,35,44,53,63,74,85,95,105,106,116,126,135,136,145,152,159,171,177,178,184,190,197,204,211,212,219,226,245,
                1,2,7,19,27,36,45,54,64,75,86,96,97,107,117,127,128,137,146,153,166,172,173,179,185,191,198,205,206,213,220,239,
                3,13,20,28,37,46,55,65,76,87,88,98,108,118,119,129,138,160,167,227,233,
                8,14,21,29,38,47,56,66,77,78,89,99,109,110,120,234,240,
                9,15,22,30,39,48,57,67,68,79,90,100,241,246,
                23,31,40,49,58,69,80,247,252
                ]

# Plotting Matrix
data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)


fig, (ax0, ax1) = plt.subplots(1, 2)
heatmap_0 = ax0.imshow(data_matrix_0, cmap='viridis', interpolation='nearest', vmin=0, vmax=50000)
heatmap_1 = ax1.imshow(data_matrix_1, cmap='viridis', interpolation='nearest', vmin=0, vmax=1500)
plt.colorbar(heatmap_0,ax=ax0)
plt.colorbar(heatmap_1, ax=ax1)


def update_heatmap(frame):
    # global data_matrix
    heatmap_0.set_data(data_matrix_0)
    heatmap_1.set_data(data_matrix_1)
    return heatmap_0, heatmap_1


def rsensor_from_adc(adc_counts: np.ndarray) -> np.ndarray:
    """
    12-bit ADC Value（sorted_one_d_array1）Converted to Crosstalk-Free Rsensor (Ω)

    —— IA Current Source（Frame-1） ——————————
      ADC Ref          VADC_REF   = 3.3   V
      IA  Ref          VREF_IA    = 0.825   V
      IA Gain          GAIN       = 1.0
      IA Postive       VDAC2      = 1.6   V
      Diode Forward    V_DIODE    = 0.46  V
      Current Source   I_OUT      = 33 µA
      Series Bias R    RS_OFFSET  = 15 kΩ
      ADC=0 Fill       RS_INVALID = 50 kΩ
    """

    # 0) Vout --------------------------------------------------
    ADC_MAX    = (1 << 12) - 1                      # 4095
    VADC_REF   = 3.3
    vout = adc_counts * VADC_REF / ADC_MAX

    # 1) ΔV ---------------------------------------------
    VREF_IA = 0.825
    GAIN    = 1.0
    delta_v = (vout - VREF_IA) / GAIN

    # 2) Limit (≤ –0.8 V)   -----------------------------------
    # SAT_NEG = -0.8
    # delta_v_mag = np.abs(np.clip(delta_v, SAT_NEG, None))

    # 3) Rsensor -----------------------------------------------
    VDAC2     = 1.6
    V_DIODE   = 0.46
    I_OUT     = 33e-6
    RS_OFFSET = 15_000.0

    rs = (VDAC2 - V_DIODE - delta_v ) / I_OUT - RS_OFFSET
    # rs = (VDAC2 - V_DIODE - delta_v_mag) / I_OUT - RS_OFFSET

    # 4) 特例：ADC 读 0 → 50 kΩ，且裁掉负值 ---------------------------
    RS_INVALID = 50_000.0
    rs = np.where(vout == 0.0, RS_INVALID, rs)

    return rs


def receive_data():    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            # Convert raw data to a 32x32 matrix of decimal integers
            decimal_array = np.frombuffer(data, dtype=np.uint16).reshape(507)   # Converting to unmapped 1-d list (253)
            
            client = decimal_array[-1]
            decimal_array1 = decimal_array[0:253]
            decimal_array2 = decimal_array[254:507]
        
            if client == 0:
                sorted_one_d_array1 = np.zeros(len(decimal_array1), dtype=int)  # mapped 1-d list
                sorted_one_d_array2 = np.zeros(len(decimal_array2), dtype=int)  # mapped 1-d list
                
                for i, index in enumerate(lookup_table):
                    sorted_one_d_array1[index] = decimal_array1[i]  # Frame-1: Gain 1.0, Vdac2 1.6V
                    
                for i, index in enumerate(lookup_table):
                    sorted_one_d_array2[index] = decimal_array2[i]  # Frame-1: Gain 16.0, Vdac2 0.75V

                Rsensor_array = rsensor_from_adc(sorted_one_d_array1) # Crosstalk-Free Rsensor (Ω) for Frame-1
            
                # Passing through Rsensor_array to sorted_one_d_array1
                # sorted_one_d_array = sorted_one_d_array1 + sorted_one_d_array2 # Plot for Original Heatmap
                sorted_one_d_array =  Rsensor_array

                # Mapping sorted 1d list to 33*11 matrix
                indices = np.argwhere(image_pattern == 1)
                for (i, j), value in zip(indices, sorted_one_d_array):
                    if value > 0:  # Adjust Min value accepted
                        data_matrix_0[i, j] = value 
                    else:
                        data_matrix_0[i, j] = 0
            
                print(data_matrix_0)

            # Right Image
            if client == 1:
                sorted_one_d_array = np.zeros(len(decimal_array), dtype=int)  # mapped 1-d list
                for i, index in enumerate(lookup_table):
                    sorted_one_d_array[index] = decimal_array[i]

                # Mapping sorted 1d list to 33*11 matrix
                indices = np.argwhere(image_pattern == 1)
                for (i, j), value in zip(indices, sorted_one_d_array):
                    if value > 200:  # Adjust Min value accepted
                        data_matrix_1[i, 14-j] = value
                    else:
                        data_matrix_1[i, 14-j] = 0

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sock.close()


if __name__ == "__main__":
    data_thread = Thread(target=receive_data)
    data_thread.daemon = True
    data_thread.start()

    ani = animation.FuncAnimation(fig, update_heatmap, interval=20, blit=True, cache_frame_data=False)
    plt.show()



