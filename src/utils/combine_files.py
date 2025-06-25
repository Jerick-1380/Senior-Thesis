def combine_python_files(file_list, output_file='combined_files_ouput.txt'):
    with open(output_file, 'w') as outfile:
        for file_name in file_list:
            try:
                with open(file_name, 'r') as infile:
                    outfile.write(f"# ===== Start of {file_name} =====\n")
                    outfile.write(infile.read())
                    outfile.write(f"\n# ===== End of {file_name} =====\n\n")
            except FileNotFoundError:
                print(f"Warning: {file_name} not found and will be skipped.")
    print(f"Combined file written to {output_file}")

# Example usage
if __name__ == '__main__':
    files = ['small_sim.py','helpers/bots.py','helpers/conversation.py','helpers/model.py'] 
    combine_python_files(files)