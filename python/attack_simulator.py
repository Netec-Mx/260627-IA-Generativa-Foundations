import os
import sys
import time

def attack_simulator():
    time.sleep(5)
    iterations = 6
    for i in range(iterations):
        print("Algo no está yendo bien...")
        sys.stdout.flush()
        time.sleep(5)
    print("⚠️Integridad del mensaje comprometida: hash no coincide.⚠️")
    print("❌Alteración detectada, tu sesión ha sido bloqueada por amenaza.❌")
    sys.exit(1)

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    while True:
        user_input = input("Enter the prompt (or type 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        attack_simulator()

if __name__ == "__main__":
    main()