import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio

def strengthToColor(strength):
    match strength:
        case 1:
            return "red"
        case 2:
            return "orange"
        case 4:
            return "blue"
        case 5:
            return "purple"
        case _:
            return "yellow"
        

class EdgeDistribution:
    def __init__(self, vertices):
        if not os.path.exists('edgedistribution'):
            os.makedirs('edgedistribution')
        self.vertices = vertices
        self.matrix = np.zeros((vertices, vertices))
        self.images = []
    def addFrame(self, dict):
        for src, dests in dict.items():
            for dest, count in dests.items():
                self.matrix[src-1, dest-1] = count
        fig, ax = plt.subplots()
        cax = ax.matshow(self.matrix, cmap='Blues')
        # Add color bar
        fig.colorbar(cax)

        # Set labels
        ax.set_xlabel('Destination Vertex')
        ax.set_ylabel('Source Vertex')
        ax.set_xticks(range(self.vertices))
        ax.set_yticks(range(self.vertices))
        ax.set_xticklabels(range(1, self.vertices+1))
        ax.set_yticklabels(range(1,self.vertices+1))
        ax.set_title('2D Histogram of Edge Occurrences')
        frame_file = f'edgedistribution/frame_{len(self.images)}.png'
        plt.savefig(frame_file)
        plt.close()
        self.images.append(imageio.imread(frame_file))
    def exportGIF(self, name, duration):
        filename = name + ".gif"
        counter = 1
        while os.path.exists(filename):
            filename = f"{name}_{counter}.gif"
            counter += 1

       # print(f"GIF exported as {filename}")
        imageio.mimsave(filename, self.images, duration=duration)



        
class Graph:
    def __init__(self, agents):
        if not os.path.exists('frames'):
            os.makedirs('frames')
        G = nx.Graph()
        for agent in agents:
            G.add_node(agent.name, color=strengthToColor(agent.strength))
        pos = nx.spring_layout(G)
        self.G = G
        self.pos = pos
        self.all_edges = []
        self.new_edges = []
        self.images = []

    def update(self, agentA, agentB):
        self.G.nodes[agentA.name]['color'] = strengthToColor(agentA.strength)
        self.G.nodes[agentB.name]['color'] = strengthToColor(agentB.strength)
        self.G.add_edge(agentA.name, agentB.name)
        self.all_edges.append((agentA.name, agentB.name))
        self.new_edges.append((agentA.name, agentB.name))

    def addFrame(self):
        plt.figure(figsize=(14, 14))
        colors = [self.G.nodes[i]['color'] for i in self.G.nodes]
        nx.draw_networkx_nodes(self.G, self.pos, node_color=colors, node_size=1000)
        nx.draw_networkx_labels(self.G, self.pos)
        if len(self.all_edges) > 0:
            old_edges = self.all_edges[:-len(self.new_edges)]  # All except the new edges
            nx.draw_networkx_edges(self.G, self.pos, edgelist=old_edges, edge_color='black')
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.new_edges, edge_color='lime', style='dashed', width=2, arrows=True
            )

        frame_file = f'frames/frame_{len(self.images)}.png'
        plt.savefig(frame_file)
        plt.close()
        self.images.append(imageio.imread(frame_file))
        self.new_edges = []  # Clear new edges after adding frame

    def exportGIF(self, name, duration):
        filename = name + ".gif"
        counter = 1
        while os.path.exists(filename):
            filename = f"{name}_{counter}.gif"
            counter += 1

       # print(f"GIF exported as {filename}")
        imageio.mimsave(filename, self.images, duration=duration)



class Writer:
    def __init__(self, agents, matrix, output_folder = "output_images"):
        self.agents = agents
        self.output_folder = output_folder
        self.matrix = matrix
        if not os.path.exists(output_folder):
            os.makedirs(output_folder )
            print(f"Folder created: {output_folder}")
    def output_desc(self, filename="desc.txt"):
        filepath = os.path.join(self.output_folder, filename)
        left = 0
        mid = 0
        right = 0
        for agent in self.agents:
            if(agent.strength<0.4):
                left+=1
            elif(agent.strength<0.6):
                mid+=1
            else:
                right+=1
        
        with open(filepath, 'w') as file:
            file.write(f"There are {left} agents who believe in con\n")  # First line: mean
            file.write(f"There are {right} agents who believe in pro\n")  # Second line: max and min
            file.write(f"There are {mid} agents who are neutral \n\n")  # Third line: variance
            for i, row in enumerate(self.matrix):
                file.write(f"Iteration {i+1}: {row}\n")  # Third line: variance



class Grapher:
    def __init__(self, matrix, output_folder="output_images"):
        """
        Initialize the visualizer with an output folder for images.
        
        Args:
            output_folder (str): Folder to store generated images.
        """
        self.matrix = matrix
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def create_gif(self, gif_name="output.gif", duration=10):
        """
        Create a GIF from a 2D list of values where each inner list contributes an image with circular layout.
        
        Args:
            value_sequences (list of lists): A 2D list where each inner list contains numbers between 0 and 1.
            gif_name (str): The name of the output GIF file.
            duration (float): Duration of each frame in the GIF (in seconds).
        """
        value_sequences = self.matrix
        if not all(all(0 <= val <= 1 for val in values) for values in value_sequences):
            raise ValueError("All values in all lists must be between 0 and 1.")
        
        # Create a directory for temporary frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_files = []

        cmap = cm.get_cmap('coolwarm')  # Colormap from red to blue

        # Use a circular layout to ensure consistency
        num_nodes = len(value_sequences[0])
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        pos = nx.circular_layout(G)

        for idx, values in enumerate(value_sequences):
            # Create the graph for the current values
            node_colors = [cmap(val) for val in values]
            
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))
            nx.draw(
                G, pos, ax=ax, node_color=node_colors, with_labels=True,
                edge_color='gray', font_weight='bold', node_size=800
            )
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
            sm.set_array([])  # Required for colorbar to function
            cbar = plt.colorbar(sm, ax=ax, label="Node Value (0=Red, 1=Blue)")
            plt.title(f"Frame {idx + 1}")
            
            # Save the frame as an image
            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
            plt.savefig(frame_path, format="png", dpi=300)
            frame_files.append(frame_path)
            plt.close(fig)
        
        # Combine the frames into a GIF
        output_path = os.path.join(self.output_folder, gif_name)
        with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
            for frame_file in frame_files:
                writer.append_data(imageio.imread(frame_file))
        
        # Clean up temporary files
        for frame_file in frame_files:
            os.remove(frame_file)
        os.rmdir(temp_dir)

        print(f"GIF created: {output_path}")
    
    def plot_lines(self, num=5, title="Line Plot Over Time", xlabel="Time", ylabel="Value", save_as="lines.png"):
        """
        Plots each column of a 2D list (matrix) as a line plot over time.

        Args:
            matrix (list of lists): A 2D list where each column represents a series of values over time.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_as (str, optional): Filename to save the plot. If None, the plot is not saved.
        """
        matrix = self.matrix
        # Ensure the matrix is not empty
        if not matrix or not matrix[0]:
            raise ValueError("The matrix must contain at least one row and one column.")

        # Convert the matrix to a numpy array for easier slicing
        data = np.array(matrix)
        num_rows, num_columns = data.shape

        # Plot each column as a line
        plt.figure(figsize=(10, 6))
        random_columns = np.random.choice(num_columns, size=num, replace=False)
        for col_idx in random_columns:
            plt.plot(range(num_rows), data[:, col_idx], label=f"Column {col_idx + 1}")

        # Add labels, title, legend, and grid
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        # Save the plot if a filename is provided
        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        # Show the plot
        plt.show()
    def plot_mean(self, title="Mean Value Over Time", xlabel="Time", ylabel="Mean Value", save_as="means.png"):
        """
        Plots the mean value of each row (over all columns) over time.

        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_as (str, optional): Filename to save the plot. If None, the plot is not saved.
        """
        mean_values = np.mean(self.matrix, axis=1)

        # Plot the mean values
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(mean_values)), mean_values, label="Mean", color="blue")

        # Add labels, title, legend, and grid
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        # Save the plot if a filename is provided
        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        # Show the plot
        plt.show()

    def plot_variance(self, title="Variance Over Time", xlabel="Time", ylabel="Variance", save_as="variances.png"):
        """
        Plots the variance of each row (over all columns) over time.

        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_as (str, optional): Filename to save the plot. If None, the plot is not saved.
        """
        variance_values = np.var(self.matrix, axis=1)

        # Plot the variance values
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(variance_values)), variance_values, label="Variance", color="red")

        # Add labels, title, legend, and grid
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        # Save the plot if a filename is provided
        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        # Show the plot
        plt.show()