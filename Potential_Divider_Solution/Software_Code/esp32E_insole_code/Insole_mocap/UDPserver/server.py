import socket

# UDP Configuration
UDP_IP = "192.168.137.1"  # IP address to bind to
UDP_PORT = 44444          # Port to listen on

# Buffer size for one frame of ADC values (255 values, each 2 bytes)
BUFFER_SIZE = 250

def main():
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

    try:
        while True:
            # Receive data from socket
            data, addr = sock.recvfrom(BUFFER_SIZE*16)
            adc_values = [int.from_bytes(data[i:i + 2], byteorder='big') for i in range(1, len(data)-9, 2)]
            print(len(adc_values))
            print(adc_values)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
