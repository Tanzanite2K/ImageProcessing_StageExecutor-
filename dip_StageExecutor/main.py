from stages import (
    acquisition, enhancement, restoration, morphology,
    segmentation, representation, recognition, compression, color_processing
)

def menu():
    print("""
    DIP Stage Executor:
    1. Image Acquisition
    2. Image Enhancement
    3. Image Restoration
    4. Morphological Processing
    5. Segmentation
    6. Representation and Description
    7. Object Recognition
    8. Image Compression
    9. Color Image Processing
    """)
    choice = int(input("Enter the DIP stage number (1-9): "))
    return choice

def execute(choice):
    if choice == 1:
        acquisition.run()
    elif choice == 2:
        enhancement.run()
    elif choice == 3:
        restoration.run()
    elif choice == 4:
        morphology.run()
    elif choice == 5:
        segmentation.run()
    elif choice == 6:
        representation.run()
    elif choice == 7:
        recognition.run()
    elif choice == 8:
        compression.run()
    elif choice == 9:
        color_processing.run()
    else:
        print("Invalid input.")

if __name__ == "__main__":
    user_choice = menu()
    execute(user_choice)
