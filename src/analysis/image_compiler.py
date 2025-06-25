from PIL import Image
import os

def combine_images_from_folders(folder_names, grid_layout, image_name=None):
    """
    Combine images from specified folders into a single grid image.

    Parameters:
    folder_names (list): List of folder paths containing images
    grid_layout (tuple): A tuple of (rows, columns) specifying the grid layout
    image_name (str, optional): Specific image name to look for in each folder. If None, selects the first image.

    Returns:
    PIL.Image: A combined image with all images arranged in the specified grid
    """
    # Validate input
    rows, cols = grid_layout
    if len(folder_names) > rows * cols:
        raise ValueError(f"Too many folders for the specified grid layout {grid_layout}")

    images = []
    max_width, max_height = 0, 0

    for folder in folder_names:
        img_path = None

        if image_name:
            # Check if the specific image exists in the folder
            potential_path = os.path.join(folder, image_name)
            if os.path.exists(potential_path):
                img_path = potential_path
        else:
            # Get the first image in the folder if no specific image name is given
            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            if image_files:
                img_path = os.path.join(folder, image_files[0])

        if img_path is None:
            raise ValueError(f"No suitable image found in folder: {folder}")

        img = Image.open(img_path)
        images.append(img)

        # Track maximum image dimensions
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)

    # Create a blank image for the grid
    grid_width = max_width * cols
    grid_height = max_height * rows
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Place images in the grid
    for index, img in enumerate(images):
        row, col = divmod(index, cols)

        # Resize image to match max dimensions while maintaining aspect ratio
        img_resized = img.resize((max_width, max_height), Image.LANCZOS)

        # Calculate position
        x, y = col * max_width, row * max_height

        # Paste the image
        grid_image.paste(img_resized, (x, y))

    return grid_image

# Example usage
def main():
    folders = [
        'output5020', 
        'output5021', 
        'output5022',
        'output5023', 
        'output5024'
    ]
    
    grid = (1, 5)

    # Use a specific image name (or None to take the first available image)
    image_name = "means.png"

    combined_image = combine_images_from_folders(folders, grid, image_name)
    combined_image.save('means_god_none.png')

if __name__ == '__main__':
    main()