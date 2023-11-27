import multiprocessing

def update_shared_dict(shared_dict, key, value):
    # Update the shared dictionary
    shared_dict[key] = value

if __name__ == "__main__":
    # Create a multiprocessing manager
    with multiprocessing.Manager() as manager:
        # Create a shared dictionary using the manager
        shared_dict = manager.dict()

        # Define the list of arguments for starmap
        arguments_list = [
            (shared_dict, 'key1', 'value1'),
            (shared_dict, 'key2', 'value2'),
            # Add more arguments as needed
        ]

        # Create a multiprocessing pool
        with multiprocessing.Pool() as pool:
            # Use starmap to update the shared dictionary with multiple processes
            pool.starmap(update_shared_dict, arguments_list)

        # Print the updated shared dictionary
        print("Updated Shared Dictionary:", shared_dict)
