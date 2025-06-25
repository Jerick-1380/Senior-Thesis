import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
import os
import numpy as np
import json
import csv

def strengthToColor(strength):
    """
    Convert a numeric strength value to a predefined color code.
    """
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
    """
    Tracks occurrences of directed edges between vertices 
    and creates frames for visualization.
    """

    def __init__(self, vertices):
        if not os.path.exists('edgedistribution'):
            os.makedirs('edgedistribution')
        self.vertices = vertices
        self.matrix = np.zeros((vertices, vertices))
        self.images = []

    def addFrame(self, edge_dict):
        """
        Update the internal matrix with new edge occurrences from edge_dict 
        and save a frame representing the distribution.
        """
        for src, dests in edge_dict.items():
            for dest, count in dests.items():
                self.matrix[src-1, dest-1] = count

        fig, ax = plt.subplots()
        cax = ax.matshow(self.matrix, cmap='Blues')
        fig.colorbar(cax)
        ax.set_xlabel('Destination Vertex')
        ax.set_ylabel('Source Vertex')
        ax.set_xticks(range(self.vertices))
        ax.set_yticks(range(self.vertices))
        ax.set_xticklabels(range(1, self.vertices + 1))
        ax.set_yticklabels(range(1, self.vertices + 1))
        ax.set_title('2D Histogram of Edge Occurrences')

        frame_file = f'edgedistribution/frame_{len(self.images)}.png'
        plt.savefig(frame_file)
        plt.close()
        self.images.append(imageio.imread(frame_file))

    def exportGIF(self, name, duration):
        """
        Export the collected frames as a GIF.
        """
        filename = name + ".gif"
        counter = 1
        while os.path.exists(filename):
            filename = f"{name}_{counter}.gif"
            counter += 1
        imageio.mimsave(filename, self.images, duration=duration)


class Graph:
    """
    Manages a network of Agents and their edges, producing frames for a GIF.
    """

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
        """
        Update the graph with new edges and node colors 
        based on the agents' strengths.
        """
        self.G.nodes[agentA.name]['color'] = strengthToColor(agentA.strength)
        self.G.nodes[agentB.name]['color'] = strengthToColor(agentB.strength)
        self.G.add_edge(agentA.name, agentB.name)
        self.all_edges.append((agentA.name, agentB.name))
        self.new_edges.append((agentA.name, agentB.name))

    def addFrame(self):
        """
        Save a frame of the current graph state.
        Newly added edges since last frame are highlighted differently.
        """
        plt.figure(figsize=(14, 14))
        colors = [self.G.nodes[node]['color'] for node in self.G.nodes]
        nx.draw_networkx_nodes(self.G, self.pos, node_color=colors, node_size=1000)
        nx.draw_networkx_labels(self.G, self.pos)

        if len(self.all_edges) > 0:
            old_edges = self.all_edges[:-len(self.new_edges)]
            nx.draw_networkx_edges(self.G, self.pos, edgelist=old_edges, edge_color='black')
            nx.draw_networkx_edges(
                self.G, 
                self.pos, 
                edgelist=self.new_edges, 
                edge_color='lime', 
                style='dashed', 
                width=2, 
                arrows=True
            )

        frame_file = f'frames/frame_{len(self.images)}.png'
        plt.savefig(frame_file)
        plt.close()
        self.images.append(imageio.imread(frame_file))
        self.new_edges = []

    def exportGIF(self, name, duration):
        """
        Export the collected frames into a single GIF file.
        """
        filename = name + ".gif"
        counter = 1
        while os.path.exists(filename):
            filename = f"{name}_{counter}.gif"
            counter += 1
        imageio.mimsave(filename, self.images, duration=duration)


class Writer:
    """
    Collects descriptions about Agents and outputs them to a file.
    Also stores iteration matrix data.
    """

    def __init__(self, agents, matrix, output_folder="output_images"):
        self.agents = agents
        self.output_folder = output_folder
        self.matrix = matrix
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Folder created: {output_folder}")

    def output_desc(self, filename="desc.txt"):
        """
        Write some aggregated agent info and iteration states to a text file.
        """
        filepath = os.path.join(self.output_folder, filename)
        left = 0
        mid = 0
        right = 0
        for agent in self.agents:
            if agent.strength < 0.4:
                left += 1
            elif agent.strength < 0.6:
                mid += 1
            else:
                right += 1

        with open(filepath, 'w') as file:
            file.write(f"There are {left} agents who believe in con\n")
            file.write(f"There are {right} agents who believe in pro\n")
            file.write(f"There are {mid} agents who are neutral \n\n")
            for i, row in enumerate(self.matrix):
                file.write(f"Iteration {i+1}: {row}\n")
            variance_values = np.var(np.array(self.matrix), axis=1)
            convergence_rate = []
            for i in range(1, len(self.matrix)):
                diff = np.linalg.norm(np.array(self.matrix[i]) - np.array(self.matrix[i-1]))
                convergence_rate.append(diff)
            file.write(f"Variances: {variance_values}\n")
            file.write(f"Convergence Rates: {convergence_rate}\n")
                
    def write_csv_log(self, conversations, filename="simulation_log.csv"):
        """
        Write a structured CSV log of each conversation step.
        
        Args:
            conversations (list): A list of conversation records. Each record
                could be a dict like:
                {
                    "round": int,
                    "agentA": {"name": ..., "strength": ..., "off": ..., "args": [...]},
                    "agentB": {"name": ..., "strength": ..., "off": ..., "args": [...]},
                    "prompt": "...",
                    "conversation_text": "... (optional)"
                    ...
                }
            filename (str): The name of the CSV file to create.
        """
        csv_path = os.path.join(self.output_folder, filename)
        
        # Define column headers; adapt them to your actual conversation data
        fieldnames = [
            "round",
            "agentA_name",
            "agentA_strength",
            "agentA_off",
            "agentA_args",
            "agentB_name",
            "agentB_strength",
            "agentB_off",
            "agentB_args",
            "prompt"
        ]

        # If you also store the raw conversation text or other metadata,
        # you can add them to 'fieldnames' above.

        with open(csv_path, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in conversations:
                row = {
                    "round": record.get("round"),
                    "agentA_name": record["agentA"]["name"],
                    "agentA_strength": record["agentA"].get("strength", ""),
                    "agentA_off": record["agentA"].get("off", ""),
                    "agentA_args": record["agentA"].get("args", ""),
                    "agentB_name": record["agentB"]["name"],
                    "agentB_strength": record["agentB"].get("strength", ""),
                    "agentB_off": record["agentB"].get("off", ""),
                    "agentB_args": record["agentB"].get("args", ""),
                    "prompt": record.get("prompt", "")
                }
                writer.writerow(row)

        print(f"CSV log written to {csv_path}")

    def write_conversation_summaries_json(self, conversations, filename="conversation_summaries.json"):
        """
        Write detailed conversation summaries to a JSON file.

        Args:
            conversations (list): A list of conversation summaries. Each item
                might contain:
                {
                    "round": int,
                    "agentA_name": str,
                    "agentB_name": str,
                    "initial_prompt": str,
                    "final_strengths": (float, float),
                    "key_exchanges": [ ... ],
                    ...
                }
            filename (str): The name of the JSON file to create.
        """
        json_path = os.path.join(self.output_folder, filename)

        # Each element in 'conversations' can be any custom structure you want
        # as long as it's JSON-serializable.

        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(conversations, jsonfile, indent=2, ensure_ascii=False)

        print(f"JSON summaries written to {json_path}")

class Grapher:
    """
    Creates visual representations (plots/GIFs) from a 2D numeric matrix over time.
    """

    def __init__(self, matrix, output_folder="output_images"):
        self.matrix = matrix
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def create_gif(self, gif_name="output.gif", duration=10):
        """
        Create a GIF where each row in self.matrix is rendered as 
        node colors in a circular layout.
        """
        value_sequences = self.matrix
        if not all(all(0 <= val <= 1 for val in values) for values in value_sequences):
            raise ValueError("All values in all lists must be between 0 and 1.")

        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_files = []

        cmap = cm.get_cmap('coolwarm')
        num_nodes = len(value_sequences[0]) if len(value_sequences) > 0 else 0
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        pos = nx.circular_layout(G)

        for idx, values in enumerate(value_sequences):
            node_colors = [cmap(val) for val in values]

            fig, ax = plt.subplots(figsize=(8, 6))
            nx.draw(
                G, 
                pos, 
                ax=ax, 
                node_color=node_colors, 
                with_labels=True,
                edge_color='gray', 
                font_weight='bold', 
                node_size=800
            )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Node Value (0=Red, 1=Blue)")
            plt.title(f"Frame {idx + 1}")

            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
            plt.savefig(frame_path, format="png", dpi=300)
            frame_files.append(frame_path)
            plt.close(fig)

        output_path = os.path.join(self.output_folder, gif_name)
        with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
            for frame_file in frame_files:
                writer.append_data(imageio.imread(frame_file))

        for frame_file in frame_files:
            os.remove(frame_file)
        os.rmdir(temp_dir)
        print(f"GIF created: {output_path}")

    def plot_lines(self, num=5, title="Line Plot Over Time", xlabel="Time", ylabel="Value", save_as="lines.png"):
        """
        Plot each column in self.matrix as a line over the row index (time).
        Only plot up to `num` randomly selected columns if the matrix has many columns.
        """
        data = np.array(self.matrix)
        num_rows, num_columns = data.shape

        plt.figure(figsize=(10, 6))
        random_columns = np.random.choice(num_columns, size=num, replace=False)
        for col_idx in random_columns:
            plt.plot(range(num_rows), data[:, col_idx], label=f"Column {col_idx + 1}")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        plt.show()

    def plot_mean(self, title="Mean Value Over Time", xlabel="Time", ylabel="Mean Value", save_as="means.png"):
        """
        Plot the mean of each row in self.matrix.
        """
        mean_values = np.mean(self.matrix, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(mean_values)), mean_values, label="Mean", color="blue")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        plt.show()
        
    def plot_mean_with_rolling(self, k=20, title="Mean Value Over Time (With Rolling Average)",
                           xlabel="Time", ylabel="Mean Value", save_as="means_with_rolling.png"):
        """
        Plot the mean of each row in self.matrix and overlay a rolling (moving) 
        average of that mean with a window size 'k'.
        
        This version does not fill any initial gaps: the rolling average starts
        only after k data points are available, so the first rolling average 
        point is at index k-1.
        
        Args:
            k (int): The window size for the rolling average.
            title (str): Plot title.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
            save_as (str): Filename for saving the plot image.
        """
        mean_values = np.mean(self.matrix, axis=1)

        # Use 'valid' mode to avoid partial windows. 
        # Length of 'rolling_means' = len(mean_values) - k + 1
        rolling_means = np.convolve(mean_values, np.ones(k)/k, mode='valid')
        
        # The x-values for the rolling means should start at index k-1
        x_rolling = np.arange(k-1, len(mean_values))

        plt.figure(figsize=(10, 6))

        # Plot the raw mean
        plt.plot(range(len(mean_values)), mean_values, label="Mean", color="blue")

        # Plot the rolling average starting from x = k-1
        plt.plot(x_rolling, rolling_means, label=f"Rolling Mean (k={k})", color="red")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        # Save the plot if 'save_as' is specified
        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        plt.show()


    def plot_variance(self, title="Variance Over Time", xlabel="Time", ylabel="Variance", save_as="variances.png"):
        """
        Plot the variance of each row in self.matrix.
        """
        variance_values = np.var(self.matrix, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(variance_values)), variance_values, label="Variance", color="red")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        plt.show()
        
    def plot_convergence(self, title="Convergence Over Time", xlabel="Time", ylabel="Change in Norm", save_as="convergence.png"):
        """
        Plot the convergence in self.matrix.
        """
        convergence_rate = []
        for i in range(1, len(self.matrix)):
            diff = np.linalg.norm(np.array(self.matrix[i]) - np.array(self.matrix[i-1]))
            convergence_rate.append(diff)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(convergence_rate)), convergence_rate, label="L2 Difference", color="red")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)

        if save_as:
            full_path = os.path.join(self.output_folder, save_as)
            plt.savefig(full_path, format="png", dpi=300)
            print(f"Plot saved as {full_path}")

        plt.show()
        
        
        
        