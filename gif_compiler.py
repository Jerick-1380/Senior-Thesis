import os
from PIL import Image, ImageSequence
import argparse

def create_grid_gif_from_folder(folder_path, m, n, output_path):
    # Get all gif files from the specified folder
    gif_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.gif')]
    
    # Sort the list to maintain a consistent order
    gif_list.sort()

    # Load the gifs
    gifs = [Image.open(gif) for gif in gif_list]
    
    if not gifs:
        print("No GIFs found in the specified folder.")
        return
    
    # Get the size of each gif (assuming all gifs are of the same size)
    gif_width, gif_height = gifs[0].size
    
    # Create a list to store each frame of the final gif
    frames = []

    # Determine the maximum number of frames across all GIFs
    max_frames = max(gif.n_frames for gif in gifs)

    # Iterate through each frame index
    for frame_index in range(max_frames):
        # Create a new blank image for each frame
        grid_image = Image.new('RGBA', (n * gif_width, m * gif_height))
        
        # Paste each GIF frame into the correct position in the grid
        for i, gif in enumerate(gifs):
            try:
                gif.seek(frame_index)  # Move to the current frame in each gif
            except EOFError:
                gif.seek(gif.n_frames - 1)  # If the GIF is out of frames, use the last frame

            row = i // n
            col = i % n
            position = (col * gif_width, row * gif_height)
            grid_image.paste(gif, position)
        
        # Append the grid frame to the frames list
        frames.append(grid_image)

    # Save the combined image as a gif with all frames
    frames[0].save(output_path, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=gifs[0].info['duration'])

# Example usage:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--s', type=str, default = "output_grid", help='Choices to give Agent, -1 gives all')
    parser.add_argument('--m', type=int, help='Choices to give Agent, -1 gives all')
    parser.add_argument('--n', type=int, help='Choices to give Agent, -1 gives all')
    args = parser.parse_args()
    folder_path = "output2"
    m, n = args.m, args.n
    output_path = f"{args.s}.gif"

    create_grid_gif_from_folder(folder_path, m, n, output_path)