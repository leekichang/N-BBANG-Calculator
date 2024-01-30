import copy
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

def generate_graph(users):
    n_user = len(users)
    adj_mat = np.zeros((n_user,n_user)).astype(np.int64)
    return adj_mat

def N_BBANG(expense, targets):
    n_target = len(targets)
    return int(expense/n_target)

def update_graph(adj_mat, expense, spender, targets, users):
    for target in targets:
        t_idx = users.index(target)
        s_idx = users.index(spender)
        adj_mat[t_idx][s_idx] += expense
    return adj_mat

def remove_self_edge(adj_mat):
    for i in range(len(adj_mat)):
        adj_mat[i,i]=0
    return adj_mat

def remove_exchanges(adj_mat):
    for i in range(len(adj_mat)):
        for j in range(i+1, len(adj_mat)):
            to_send = adj_mat[i, j]
            to_receive = adj_mat[j, i]
            if to_send >= to_receive:
                adj_mat[i,j] -= to_receive
                adj_mat[j,i]  = 0
            else:
                adj_mat[i,j]  = 0
                adj_mat[j,i] -= to_send
    return adj_mat

def reduce_edges(adj_mat):
    '''
    1) 내(A)가 보낼 돈이 있는지 확인
    2) 내가 돈을 받을 사람(B) 중에 내가 보내야 하는 사람(C)에게 줄 돈이 있는 사람이 있는지 확인
    3) 내가 보내야 하는 돈 보다 B에게 받을 돈이 많은지 확인
    4) 내가 보내야 하는 돈을 B에게 감면해주고 C에게 추가로 보내도록
    5) C가 내야 하는 돈 중에 내가 대신 내줄 수 있는 돈이 있는지 확인
    6) 변화가 없을 때까지 반복
    '''
    last_adj_mat = np.zeros(adj_mat.shape)
    count = 0
    while not np.array_equal(last_adj_mat, adj_mat):
        last_adj_mat = copy.deepcopy(adj_mat)
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat)):
                if adj_mat[i,j] != 0:
                    for k in range(i+1, len(adj_mat)):
                        if adj_mat[k, i] > adj_mat[i,j] and adj_mat[k,j] != 0:
                            adj_mat[k,i]-=adj_mat[i,j]
                            adj_mat[k,j]+=adj_mat[i,j]
                            adj_mat[i,j] = 0
                            count += 1
                            print(f"REDUCING EDGE {count}, {np.count_nonzero(adj_mat)}")
                            break
                    for k in range(len(adj_mat)):
                        if adj_mat[j, k] > adj_mat[i,j] and adj_mat[i,k] !=0:
                            adj_mat[i,k] += adj_mat[i,j]
                            adj_mat[j,k] -= adj_mat[i,j]
                            adj_mat[i,j] = 0
                            count += 1
                            print(f"REDUCING EDGE {count}, {np.count_nonzero(adj_mat)}")
                            break
    return last_adj_mat

df = pd.read_excel('./bottom_line.xlsx', engine='openpyxl')
users = df['전체 인원'].dropna().to_list()
adj_mat = generate_graph(users)

for index in range(len(df)):
    row        = df.iloc[index]
    targets    = copy.deepcopy(users)
    spender    = row['지출인']
    expense    = row['금액']
    exceptions = row['N빵 제외자']
    if exceptions is not np.nan:
        try:
            exceptions = exceptions.split(',')
        except:
            exceptions = list(exceptions)
        for e in exceptions:
            targets.remove(e)
    
    money_to_send = N_BBANG(expense, targets)
    adj_mat = update_graph(adj_mat, money_to_send, spender, targets, users)
adj_mat = remove_self_edge(adj_mat)
# df_cm = pd.DataFrame(adj_mat, index=users, columns=users)
# sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()

G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

node_names = {i:users[i] for i in range(len(users))}
G = nx.relabel_nodes(G, node_names)

pos = nx.kamada_kawai_layout(G)  # 레이아웃 정의
# Edge weight 추출
edge_weights = nx.get_edge_attributes(G, 'weight')

# # 그래프와 edge weight 함께 시각화
# nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10, edge_color=edge_weights.values(), width=2, alpha=0.7)

# # Edge weight에 대한 수치 표시
# edge_labels = {(i, j): w['weight'] for (i, j, w) in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

# # 그래프 표시
# plt.title("Graph from Adjacency Matrix")
# plt.show()
original_n_edges = G.number_of_edges()
print(f"NUMBER OF NAIVE EDGES: {original_n_edges}")


adj_mat = remove_exchanges(adj_mat)
adj_mat = reduce_edges(adj_mat)

G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
num_edges = G.number_of_edges()
print(f"NUMBER OF NAIVE EDGES: {num_edges} REDUCED {original_n_edges-num_edges} EDGES")
print(adj_mat)
for idx, user in enumerate(users):
    print(f"{user} receive {sum(adj_mat[:,idx])} and send {sum(adj_mat[idx,:])}")

node_names = {i:users[i] for i in range(len(users))}
G = nx.relabel_nodes(G, node_names)

pos = nx.circular_layout(G)  # 레이아웃 정의
edge_weights = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10, edge_color=edge_weights.values(), width=2, alpha=0.7)

edge_labels = {(i, j): w['weight'] for (i, j, w) in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

plt.title("Graph from Adjacency Matrix")
plt.show()
