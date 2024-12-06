import sys
import os
from typing import Optional

def load_and_process_image(image_path: str, width: int = 100):
    """
    Load and process the image for text art generation.
    
    Args:
        image_path (str): Path to the image file
        width (int): Desired width for resizing
    
    Returns:
        PIL.Image: Processed grayscale image
    """
    from PIL import Image
    
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Open and process image
    try:
        image = Image.open(image_path)
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio * 0.5)  # Adjust for character height
        
        # Resize and convert to grayscale
        resized_image = image.resize((width, height), Image.LANCZOS).convert('L')
        return resized_image
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def get_emotion_based_char_set(emotion: str) -> list:
    """
    Get character set based on emotion.
    
    Args:
        emotion (str): Detected emotion
    
    Returns:
        list: Character set for the given emotion
    """
    EMOTION_CHAR_SETS = {
        'Happy': [" ", ".", ":", "*", "+"],
        'Sad': [".", "-", "=", "#", "@"],
        'Angry': ["#", "@", "&", "%", "+"],
        'Neutral': [" ", ".", ":", "-", "=", "+", "*", "#"]
    }
    
    # Normalize emotion input and handle case insensitivity
    emotion = emotion.capitalize()
    return EMOTION_CHAR_SETS.get(emotion, EMOTION_CHAR_SETS['Neutral'])

def get_character_for_intensity(pixel_intensity: int, char_set: list) -> str:
    """
    Map pixel intensity to a character.
    
    Args:
        pixel_intensity (int): Pixel brightness value (0-255)
        char_set (list): Set of characters to use
    
    Returns:
        str: Selected character based on intensity
    """
    index = int(pixel_intensity / 255 * (len(char_set) - 1))
    return char_set[index]

def save_text_art(text_art: str, output_path: str):
    """
    Save generated text art to a file.
    
    Args:
        text_art (str): Generated text art
        output_path (str): Path to save the text art
    """
    try:
        with open(output_path, 'w') as f:
            f.write(text_art)
    except IOError as e:
        raise IOError(f"Error saving text art: {e}")

def generate_text_art(image_path: str, width: int = 100, emotion: str = "Neutral") -> str:
    """
    Generate text art from an image file.
    
    Args:
        image_path (str): Path to the image file
        width (int): Desired width for the resized image
        emotion (str): Recognized emotion to adjust the character set
    
    Returns:
        str: The generated text art
    """
    try:
        # Load, resize, and convert the image to grayscale
        image = load_and_process_image(image_path, width)
        
        # Get the character set based on emotion
        char_set = get_emotion_based_char_set(emotion)
        
        # Generate text art
        text_art_lines = []
        for y in range(image.height):
            row = []
            for x in range(image.width):
                pixel_intensity = image.getpixel((x, y))
                char = get_character_for_intensity(pixel_intensity, char_set)
                row.append(char)
            text_art_lines.append(''.join(row))
        
        return '\n'.join(text_art_lines)
    
    except Exception as e:
        print(f"Error generating text art: {e}")
        sys.exit(1)

def main():
    """
    Main function to handle the flow of the program.
    Prompts user for inputs and generates text art.
    """
    try:
        # User input for image path and parameters
        while True:
            image_path = input("Enter the path to the image: ").strip()
            if os.path.exists(image_path):
                break
            print("Invalid image path. Please try again.")
        
        # Width input with validation
        while True:
            try:
                width_input = input("Enter the desired width for the text art (default is 100): ").strip()
                width = int(width_input) if width_input else 100
                if width > 0:
                    break
                print("Width must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")
        
        # Emotion input with validation
        valid_emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
        while True:
            emotion = input(f"Enter the emotion ({', '.join(valid_emotions)}, default is Neutral): ").strip().capitalize()
            emotion = emotion if emotion in valid_emotions else 'Neutral'
            break
        
        print(f"Processing the image: {image_path}...")
        
        # Generate text art
        text_art = generate_text_art(image_path, width, emotion)
        
        # Save the generated text art
        while True:
            output_path = input("Enter the path to save the generated text art (default is 'text_art.txt'): ").strip()
            output_path = output_path if output_path else "text_art.txt"
            
            try:
                save_text_art(text_art, output_path)
                print(f"Text art generated and saved to {output_path}")
                break
            except IOError as e:
                print(f"Error saving file: {e}. Please try a different path.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "_main_":
    main()