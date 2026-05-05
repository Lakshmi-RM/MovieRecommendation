#Import libraries
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random

#Read and Load Data from the Movie and Ratings CSV File
def load_data(ratings_path, movies_path):
  ratings = pd.read_csv(ratings_path)
  movies = pd.read_csv(movies_path)
  return ratings, movies

#Properly Remap data into consistent sequential format
#Create 2 new columns user_idx and movie_idx
def remapping_ids(df):
    user_ids = sorted(df["userId"].unique())
    movie_ids = sorted(df["movieId"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    df = df.copy()
    df["user_idx"] = df["userId"].map(user_to_idx)
    df["movie_idx"] = df["movieId"].map(movie_to_idx)
    return df, user_to_idx, movie_to_idx

#Build the networkX graph function
def build_nx_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        #Access movie data
        user_node = f"user_{int(row['user_idx'])}"
        movie_node = f"movie_{int(row['movie_idx'])}"
        G.add_node(user_node, node_type="user")
        G.add_node(movie_node, node_type="movie")
        G.add_edge(user_node, movie_node, weight=row["rating"])
    return G

#Build the PyTorch Geometric graph
def build_pyg_graph(df, num_users, num_movies):
    #Convert data to tensors
    user_idx = torch.tensor(df["user_idx"].values,dtype=torch.long)
    movie_idx = torch.tensor(df["movie_idx"].values + num_users,  dtype=torch.long)

    #Building undirected graph and add edges in both directions
    edge_index = torch.stack([
        torch.cat([user_idx, movie_idx]),
        torch.cat([movie_idx, user_idx]),
    ], dim=0)
    return edge_index

def train_test_split(df, test_ratio=0.2):
    train_rows, test_rows = [], []
    #identify every unique user
    for user_idx in df["user_idx"].unique():
        user_df = df[df["user_idx"] == user_idx].sample(frac=1, random_state=42)
        n_test = max(1, int(len(user_df) * test_ratio))
        test_rows.append(user_df.iloc[:n_test])
        train_rows.append(user_df.iloc[n_test:])
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df

#Apply the Matrix Factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super().__init__()
        #Creates two lookup tables (Embeddings)
        self.user_emb = nn.Embedding(num_users,  embed_dim)
        self.movie_emb = nn.Embedding(num_movies, embed_dim)

    #Create a score to determine how much a user likes a movie
    def score(self, user_idx, movie_idx):
        u = self.user_emb(user_idx)
        m = self.movie_emb(movie_idx)
        return (u * m).sum(dim=-1) # Dot product

#Apply the GCN Recommender model
class GCNRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super().__init__()
        total_nodes = num_users + num_movies #Combine nodes #think
        self.node_emb = nn.Embedding(num_users + num_movies, embed_dim)
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)

    #Analyze graph structure and improve embeddings
    def forward(self, edge_index):
        x = self.node_emb.weight
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(x) #think
        x = self.conv2(x, edge_index)
        return x

    #Determines score for likelihood of a match
    def score(self, embeddings, user_idx, movie_idx):
        u = embeddings[user_idx]
        m = embeddings[movie_idx]
        return (u * m).sum(dim=-1)

# NCF (Neural Collaborative Filtering) Recommender model
class NCFRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super().__init__()
        self.user_emb_gmf = nn.Embedding(num_users, embed_dim)
        self.item_emb_gmf = nn.Embedding(num_movies, embed_dim)
        self.user_emb_mlp = nn.Embedding(num_users, embed_dim)
        self.item_emb_mlp = nn.Embedding(num_movies, embed_dim)
        self.mlp_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, embed_dim))
        self.fc_final = nn.Linear(embed_dim * 2, 1)

    def score(self, user_idx, movie_idx):
        # GMF Path
        user_gmf = self.user_emb_gmf(user_idx)
        item_gmf = self.item_emb_gmf(movie_idx)
        gmf_output = user_gmf * item_gmf
        
        # MLP Path
        user_mlp = self.user_emb_mlp(user_idx)
        item_mlp = self.item_emb_mlp(movie_idx)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Concatenate GMF and MLP to form NCF
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        return self.fc_final(combined).squeeze(-1)

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

# Negative sampling (Find movies that were not watched)
def sample_negative(df, num_users, num_movies):
    neg_users, neg_movies = [], []
    #Search through every user
    for user_idx in df["user_idx"].unique():
        seen = set(df[df["user_idx"] == user_idx]["movie_idx"].values)
        unseen = [m for m in range(num_movies) if m not in seen]
        if not unseen:
            continue
        rand_i = torch.randint(len(unseen), (1,)).item()
        neg_users.append(user_idx)
        neg_movies.append(unseen[rand_i])   # raw movie idx (no offset)
    return torch.tensor(neg_users), torch.tensor(neg_movies)

#Training loop to teach GCN Model for reccomendations
def train_gcn(model, edge_index, train_df, num_users, num_movies, epochs=20, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_user_idx = torch.tensor(train_df["user_idx"].values,              dtype=torch.long)
    pos_movie_idx = torch.tensor(train_df["movie_idx"].values + num_users, dtype=torch.long)

    #Repeat every 20 times and calculate scores
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(edge_index)
        pos_scores = model.score(embeddings, pos_user_idx, pos_movie_idx)
        neg_user_idx, neg_movie_idx = sample_negative(train_df, num_users, num_movies)
        neg_movie_offset = neg_movie_idx + num_users
        neg_scores = model.score(embeddings, neg_user_idx, neg_movie_offset)
        min_len = min(len(pos_scores), len(neg_scores))
        loss = bpr_loss(pos_scores[:min_len], neg_scores[:min_len])
        loss.backward()
        optimizer.step()
        print(f"  [GCN] Epoch {epoch+1:02d}/{epochs}  Loss: {loss.item():.4f}")

#Training loop for Matrix Factorization (MF) model
def train_mf(model, train_df, num_movies, epochs=20, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_user_idx = torch.tensor(train_df["user_idx"].values,  dtype=torch.long)
    pos_movie_idx = torch.tensor(train_df["movie_idx"].values, dtype=torch.long)

    #Repeat every 20 times and calculate scores
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_scores = model.score(pos_user_idx, pos_movie_idx)
        neg_movie_idx = torch.randint(0, num_movies, (len(pos_user_idx),))
        neg_scores = model.score(pos_user_idx, neg_movie_idx)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()#reduce loss
        optimizer.step() #update parameters
        print(f"  [MF]  Epoch {epoch+1:02d}/{epochs}  Loss: {loss.item():.4f}")

#Training loop for Neural Collaborative Filtering (NCF) model
def train_ncf(model, train_df, num_movies, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_user_idx = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    pos_movie_idx = torch.tensor(train_df["movie_idx"].values, dtype=torch.long)

    #Repeat every 20 times and calculate scores
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_scores = model.score(pos_user_idx, pos_movie_idx)
        neg_movie_idx = torch.randint(0, num_movies, (len(pos_user_idx),))
        neg_scores = model.score(pos_user_idx, neg_movie_idx)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        print(f"  [NCF] Epoch {epoch+1:02d}/{epochs}  Loss: {loss.item():.4f}")

#Check if reccomended movies were in user's interested list
def recall_at_k(recommended, relevant, k):
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0 #lowest score
    return len(top_k & relevant_set) / len(relevant_set)

#Evaluation of GCN Model and determine how well the model performs
def evaluate_gcn(model, edge_index, train_df, test_df, num_users, num_movies, k=10):
    model.eval()
    recall_scores = []
    with torch.no_grad():
        embeddings = model(edge_index)
        #Test each user across the entire population
        for user_idx in test_df["user_idx"].unique():
            relevant = test_df[test_df["user_idx"] == user_idx]["movie_idx"].tolist()
            seen_in_train = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
            all_movie_nodes = torch.tensor([i + num_users for i in range(num_movies)], dtype=torch.long)
            user_nodes = torch.full((num_movies,), user_idx, dtype=torch.long)
            scores = model.score(embeddings, user_nodes, all_movie_nodes)
            #Prevents repeated old reccomendations
            for seen_idx in seen_in_train:
                scores[seen_idx] = float("-inf")
            top_indices = scores.topk(k).indices.tolist()
            recall_scores.append(recall_at_k(top_indices, relevant, k))
    return sum(recall_scores) / len(recall_scores)

#Evaluation of Matrix Factorization Model and determines how well the model performs
def evaluate_mf(model, train_df, test_df, num_movies, k=10):
    model.eval()
    recall_scores = []
    with torch.no_grad():
        #Test each user across the entire population
        for user_idx in test_df["user_idx"].unique():
            relevant = test_df[test_df["user_idx"] == user_idx]["movie_idx"].tolist()
            seen_in_train = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
            user_tensor = torch.tensor([user_idx] * num_movies, dtype=torch.long)
            all_movies = torch.arange(num_movies, dtype=torch.long)
            scores = model.score(user_tensor, all_movies)
            #Prevents repeated old reccomendations
            for seen_idx in seen_in_train:
                scores[seen_idx] = float("-inf")
            top_indices = scores.topk(k).indices.tolist()
            recall_scores.append(recall_at_k(top_indices, relevant, k))
    return sum(recall_scores) / len(recall_scores)

#Evaluation of Neural Collaborative Filtering Model and determines how well the model performs
def evaluate_ncf(model, train_df, test_df, num_movies, k=10):
    model.eval()
    recall_scores = []
    with torch.no_grad():
        #Test each user across the entire population
        for user_idx in test_df["user_idx"].unique():
            relevant = test_df[test_df["user_idx"] == user_idx]["movie_idx"].tolist()
            seen_in_train = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
            user_tensor = torch.tensor([user_idx] * num_movies, dtype=torch.long)
            all_movies = torch.arange(num_movies, dtype=torch.long)
            scores = model.score(user_tensor, all_movies)
            #Prevents repeated old reccomendations
            for seen_idx in seen_in_train:
                scores[seen_idx] = float("-inf")
            top_indices = scores.topk(k).indices.tolist()
            recall_scores.append(recall_at_k(top_indices, relevant, k))
    return sum(recall_scores) / len(recall_scores)

#Recommendation of 10 movies to user selected using GCN model
def recommend_for_user_gcn(gcn_model, edge_index, train_df, user_idx, num_users, num_movies,
                           idx_to_movie, movie_id_to_title, k=10):
    gcn_model.eval()
    with torch.no_grad():
        embeddings = gcn_model(edge_index)
        all_movie_nodes= torch.tensor([i + num_users for i in range(num_movies)], dtype=torch.long)
        user_nodes = torch.full((num_movies,), user_idx, dtype=torch.long)
        scores = gcn_model.score(embeddings, user_nodes, all_movie_nodes)

    seen = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
    for seen_idx in seen:
        scores[seen_idx] = float("-inf")

    top_indices = scores.topk(k).indices.tolist()
    _print_recommendations("GCN", top_indices, idx_to_movie, movie_id_to_title, k)
    return top_indices

#Recommendation of 10 movies to user selected using MF model
def recommend_for_user_mf(mf_model, train_df, user_idx, num_movies, idx_to_movie, movie_id_to_title, k=10):
    mf_model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * num_movies, dtype=torch.long)
        all_movies = torch.arange(num_movies, dtype=torch.long)
        scores = mf_model.score(user_tensor, all_movies)

    seen = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
    for seen_idx in seen:
        scores[seen_idx] = float("-inf")

    top_indices = scores.topk(k).indices.tolist()
    _print_recommendations("MF", top_indices, idx_to_movie, movie_id_to_title, k)
    return top_indices

#Recommendation of 10 movies to user selected using NCF model
def recommend_for_user_ncf(ncf_model, train_df, user_idx, num_movies,
                           idx_to_movie, movie_id_to_title, k=10):
    ncf_model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * num_movies, dtype=torch.long)
        all_movies = torch.arange(num_movies, dtype=torch.long)
        scores = ncf_model.score(user_tensor, all_movies)

    seen = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
    for seen_idx in seen:
        scores[seen_idx] = float("-inf")

    top_indices = scores.topk(k).indices.tolist()
    _print_recommendations("NCF", top_indices, idx_to_movie, movie_id_to_title, k)
    return top_indices

#Print the recommendations for each model
def _print_recommendations(model_name, top_indices, idx_to_movie, movie_id_to_title, k):
    print(f"\n{'─'*55}")
    print(f"  Top {k} Recommendations  [{model_name}]")
    print(f"{'─'*55}")
    for rank, idx in enumerate(top_indices, 1):
        movie_id = idx_to_movie[idx]
        title = movie_id_to_title.get(movie_id, f"[Unknown Movie ID {movie_id}]")
        print(f"  {rank:2d}. {title}")
    print(f"{'─'*55}\n")

#Get input from user on which user the recommendatiosn needs to be printed
def get_valid_user_id(user_to_idx):
    while True:
        raw = input("Enter a userId to get recommendations: ").strip()
        try:
            uid = int(raw)
            if uid in user_to_idx:
                return uid
            else:
                print(f"  ✗ userId {uid} not found. Please choose from the dataset.")
        except ValueError:
            print("  ✗ Please enter a valid integer userId.")

#Print recall value for each model
def print_results(results):
    print(f"\n{'Model':<10} {'Recall@10':>10}")
    print("─" * 22)
    for model_name, recall in results:
        print(f"{model_name:<10} {recall:>10.4f}")
    print()

def plot_bipartite_graph(G, num_users=40, num_movies=40, max_edges=200):

    #Random Sampling
    all_users = [n for n in G.nodes if n.startswith("user_")]
    all_movies = [n for n in G.nodes if n.startswith("movie_")]
    #Randomly select a subset of users and movies
    user_nodes = random.sample(all_users, min(num_users, len(all_users)))
    movie_nodes = random.sample(all_movies, min(num_movies, len(all_movies)))
    nodes_to_keep = set(user_nodes + movie_nodes)

    #Build filtered subgraph
    edges = [
        (u, v) for u, v in G.edges()
        if u in nodes_to_keep and v in nodes_to_keep
    ]

    # Limit maximum amount of edges for clean subset
    edges = edges[:max_edges]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes_to_keep)
    subgraph.add_edges_from(edges)
    #Create bipartite layout
    pos = {}

    # Place Users on left
    for i, node in enumerate(user_nodes):
        pos[node] = (0, i)
    # Place Movies on right
    for i, node in enumerate(movie_nodes):
        pos[node] = (1, i)

    # Assign colors for nodes
    colors = []
    for node in subgraph.nodes:
        if node.startswith("user_"):
            colors.append("blue")
        else:
            colors.append("red")

    fig=plt.figure(figsize=(10, 10))
    ax=fig.add_subplot(111)

    nx.draw(subgraph, pos, node_color=colors, node_size=120, edge_color="gray", alpha=0.7, with_labels=False)

    # Create Legend for Users and Movies
    plt.scatter([], [], c="blue", label="Users")
    plt.scatter([], [], c="red", label="Movies")
    plt.legend(loc="upper right", frameon=True)

    fig.suptitle("Bipartite Graph (Users vs Movies)", fontsize=14,fontweight='bold', y=0.98)
    
    ax.set_axis_off()
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    plt.show()

#Movie Rating Network Characteristics
def graph_sparsity(G, num_users, num_movies):
    actual_edges = G.number_of_edges()
    possible_edges = num_users * num_movies
    density = actual_edges / possible_edges
    sparsity = 1 - density
    return density, sparsity, actual_edges, possible_edges


def main():
    #If csv files are present in local folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_path = os.path.join(script_dir, "ratings.csv")
    movies_path  = os.path.join(script_dir, "movies.csv")

    ratings, movies = load_data(ratings_path, movies_path)
    ratings, user_to_idx, movie_to_idx = remapping_ids(ratings)

    #Convert ID's back to movie titles
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
    movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)
    
    #Get user id from user
    target_user_id  = get_valid_user_id(user_to_idx)
    target_user_idx = user_to_idx[target_user_id]
    target_node = f"user_{target_user_idx}"

    #Split data into testing and training sets
    train_df, test_df = train_test_split(ratings)
    print(f"     Train: {len(train_df)} ratings | Test: {len(test_df)} ratings") #Confirm number of total ratings
    
    G = build_nx_graph(train_df)
    edge_index = build_pyg_graph(train_df, num_users, num_movies)
    print(f"     NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Movie Rating Network Characteristics
    density, sparsity, actual, possible = graph_sparsity(G, num_users, num_movies)

    avg_user_degree = len(train_df) / num_users
    avg_movie_degree = len(train_df) / num_movies

    print(f"Total Number of Nodes: {G.number_of_nodes()} nodes")
    print(f"Total Number of Edges: {G.number_of_edges()} edges")
    print(f"Total Number of User Nodes: {num_users}")
    print(f"Total Number of Movie Nodes: {num_movies}")
    print(f"Density:  {density:.6f} ({density*100:.4f}%)")
    print(f"Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
    print(f"Average User Degree: {avg_user_degree:.2f}")
    print(f"Average Movie Degree: {avg_movie_degree:.2f}")

    # Bipartite graph
    plot_bipartite_graph(G)

    #Train all models
    results = []

    print("Matrix Factorization")
    mf_model = MatrixFactorization(num_users, num_movies, embed_dim=64)
    train_mf(mf_model, train_df, num_movies, epochs=20, lr=0.01)
    mf_recall = evaluate_mf(mf_model, train_df, test_df, num_movies, k=10)
    results.append(("MF", mf_recall))

    print("GCN Recommender")
    gcn_model = GCNRecommender(num_users=num_users, num_movies=num_movies, embed_dim=64)
    train_gcn(gcn_model, edge_index, train_df, num_users, num_movies, epochs=20, lr=0.01)
    gcn_recall = evaluate_gcn(gcn_model, edge_index, train_df, test_df,
                              num_users, num_movies, k=10)
    results.append(("GCN", gcn_recall))

    print("NCF Recommender")
    ncf_model = NCFRecommender(num_users, num_movies, embed_dim=64)
    train_ncf(ncf_model, train_df, num_movies, epochs=20, lr=0.001)
    ncf_recall = evaluate_ncf(ncf_model, train_df, test_df, num_movies, k=10)
    results.append(("NCF", ncf_recall))

    #Results
    print("Evaluation results (Recall@10):")
    print_results(results)

    #Recommendations for selected user for all models
    print(f"Recommendations for userId = {target_user_id}  (user_idx = {target_user_idx})")

    recommend_for_user_gcn(gcn_model, edge_index, train_df, target_user_idx, num_users, num_movies,
        idx_to_movie, movie_id_to_title, k=10)
    recommend_for_user_mf(mf_model, train_df, target_user_idx, num_movies, idx_to_movie, movie_id_to_title, k=10)
    recommend_for_user_ncf(ncf_model, train_df, target_user_idx, num_movies, idx_to_movie, movie_id_to_title, k=10)

if __name__ == "__main__":
    main()