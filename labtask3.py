import pandas as pd
import networkx as nx

def read_graph_from_csv(filename):
    df = pd.read_csv(filename)
    graph = nx.from_pandas_edgelist(df, 'source', 'destination', ['weight'])
    return graph

def bfs(graph, source, goal):
    # Implement Breadth-First Search
    return

def dfs(graph, source, goal):
    # Implement Depth-First Search
    return

def bestfs(graph, source, goal):
    # Implement Best-First Search
    return

def start_search():
    filename = input("Enter the CSV file name: ")
    graph = read_graph_from_csv(filename)
    
    source = input("Enter the source node: ")
    goal = input("Enter the goal node: ")
    
    print("Choose a search algorithm:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Best-First Search (BestFS)")
    
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        path, num_nodes = bfs(graph, source, goal)
        print("Path found using Breadth-First Search:", ' -> '.join(path))
        print("Nodes traced by Breadth-First Search:", num_nodes)
    elif choice == 2:
        path, num_nodes = dfs(graph, source, goal)
        print("Path found using Depth-First Search:", ' -> '.join(path))
        print("Nodes traced by Depth-First Search:", num_nodes)
    elif choice == 3:
        path, num_nodes = bestfs(graph, source, goal)
        print("Path found using Best-First Search:", ' -> '.join(path))
        print("Nodes traced by Best-First Search:", num_nodes)
    else:
        print("Invalid choice")

# Call the start_search() function to begin the search process
start_search()
