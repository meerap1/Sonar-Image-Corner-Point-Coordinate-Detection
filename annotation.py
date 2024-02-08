import cv2

def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at (x={x}, y={y})")

def main():
    # Define the image path
    image_path = r'C:\Users\meera\Sonar-Image-Corner-Point-Cordinate-Detection-\data\test\img\2.png'

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image.")
        return
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_click_event)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
