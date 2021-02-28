# 算法模板(C++)

Author: Daniel Liu
Contact me: 191240030@smail.nju.edu.cn
Course: Problem Solving. NJU

[TOC]

## 一、二分查找

#### 1. 整数二分模板

```c++
int ans = 0, l, r;
// some input
while(l <= r){
    int m = (l + r) / 2;
    if(check(m)){
        ans = m;
        l = l + 1;
    } else {
        r = r - 1;
    }
}
```

## 二、矩阵计算

### 1. Gauss消元法

```c++
// 求解线性方程组，求解成功后A的左边方阵为单位阵，最右边一列即为解
// O(n^3)
double A[MAXN][MAXN];   // A is the nX(n+1) augmented matrix

void gauss(){
    for(int j = 1 ;j <= N ;++j){  // vis col
        int max_idx = j;
        for(int i = j + 1 ;i <= N ;++i)  // careful!
            if(abs(A[i][j]) > abs(A[max_idx][j]))
                max_idx = i;
        if(max_idx != j)
            for(int k = 1 ;k <= N + 1 ;++k)
                swap(A[max_idx][k], A[j][k]);
        else
            if(A[j][j] == 0){
                error("unsolvable");
                return;
            }
        //
        double rec_ajj = A[j][j];  // careful here!
        for(int k = 1 ;k <= N + 1 ;++k)
            A[j][k] /= rec_ajj;
        //
        for(int i = 1 ;i <= N ;++i)
            if(i != j) {
                double rec_aij = A[i][j]; // careful here!
                for (int k = 1; k <= N + 1; ++k) {
                    A[i][k] -= rec_aij * A[j][k];
                }
            }
    }
}
```

## 三、线性规划

### 1. 单纯形算法

目标函数：$f=\sum_{j=1}^n A[0][j]$

第$i$个约束条件：$\sum_{j=1}^nA[i][j]\leq A[i][0]$

```c++
const double EPS = 1e-8;

int id[MAXN+MAXM] = {};
double v[MAXN] = {};
double a[MAXM][MAXN] = {};

int sgn(double x) {   // get sign of double
  if (x < -EPS) return -1;
  return x > EPS ? 1 : 0;
}

void pivot(int r, int c) {
  swap(id[n + r], id[c]);
  double x = -a[r][c];
  a[r][c] = -1;
  for (int i = 0; i <= n; ++i) a[r][i] /= x;
  for (int i = 0; i <= m; ++i)
    if (sgn(a[i][c]) && i != r) {
      x = a[i][c];
      a[i][c] = 0;
      for (int j = 0; j <= n; ++j) a[i][j] += x * a[r][j];
    }
}

int simplex() {
  /* important: revert symbols of conditions */
  for (int i = 1; i <= m; ++i) 
    for (int j = 1; j <= n; ++j) 
       a[i][j] *= -1;
  for (int i = 1; i <= n; ++i) id[i] = i;
  /* initial-simplex */
  while (true) {
    int x = 0, y = 0;
    for (int i = 1; i <= m; ++i)
      if (sgn(a[i][0]) < 0) { x = i; break; }
    if (!x) break;
    for (int i = 1; i <= n; ++i)
      if (sgn(a[x][i]) > 0) { y = i; break; }
    if (!y) return -1; // infeasible
    pivot(x, y);
  }
  /* solve-simplex */
  while (true) {
    int x = 0, y = 0;
    for (int i = 1; i <= n; ++i)
      if (sgn(a[0][i]) > 0) { x = i; break; }
    if (!x) break; // finished
    double w = 0, t = 0; bool f = true;
    for (int i = 1; i <= m; ++i)
      if (sgn(a[i][x]) < 0) {
        t = -a[i][0] / a[i][x];
        if (f || t < w) {
          w = t, y = i, f = false;
        }
      }
    if (!y) return 1; // unbounded
    pivot(y, x);
  }
  for (int i = 1; i <= n; ++i) v[i] = 0;
  for (int i = n + 1; i <= n + m; ++i) v[id[i]] = a[i - n][0];
  return 0;
}
```



## 四、图论算法

### 并查集

#### 1.链表实现

```C++
//O(n^2)
class node{
    int val;
    node *next;
    rep *s;
    node(int v, rep* r = NULL):val(v), next(it), rep(r){}
};
class rep{
  	int weight;
  	node *head, *tail;
  	rep(int w):weight(w), head(NULL), tail(NULL){}
};
node* make_set(int val){
    node* rp = new rep(1);
    node* np = new node(val, rp);
    np->s = rp;
    rp->head = rp->tail = np;
    return np;
}
rep* find_set(node* n){
    return n->rep;
}
rep* union_set(node* n1, node* n2){
    rep *r1 = find_set(n1), *r2 = find_set(n2);
    if(r1->weight < r2->weight)
        swap(r1, r2);
    r1->tail->next = r2->head;
    r1->tail = r2->tail;
    node *p = r2->head;
    while(p != NULL){
        p->rep = r1;
        p = p->next;
    }
    delete r2;
}


```

#### 2.树实现

```c++
//O(m\alpha(n))
struct node{
    int val;
    int rank;
    node* p;
    node(int v): val(v), rank(1), p(it){}
}
node* make_set(int val){
    return new node(val);
}
void link(node* x, node* y){
    if(x->rank > y->rank)
        y->p = x;
    else{
        x->p = y;
        if(x->rank == y->rank)
            y->rank ++;
    }
}
node* find_set(node* x){
    if(x->p != x)
        x->p = find_set(x->p);  // path compression
    return x->p;
}
void union_set(node* x, node* y){
    link(find_set(x), find_set(y));
}
```

### 最小生成树

#### 1. Kruskal算法

```C++
// 适合稀疏图（点多边少）
struct edge{
    int from, to, w;
}
edge edges[MAXM] = {0};

int find_set{
    ...
}

void union_set{
    ...
}

void kruskal(){
    sort(edges， edges+m, cmp);
    int ans = 0, t = 0;
    for(int i = 1 ;i <= n-1; ++i){
        while(find_set(edges[t].from) == find_set(edges[t].to))
            ++t;
        ans += edges[t].weight;
        union_set(edges[t].from, edges[t].to);
    }
}
```

#### 2. Prim算法

```c++
// 适合稠密图（点少边多）
bool vis[MAXN] = {0};
int low[MAXN]; // distance to the closest vertice in tree
vector<pair<int, int>> edge[MAXN]; // save edge and weight
int ans = 0;
void prim(){
    // initialize
    vis[1] = true;
    for(int i = 1 ;i <= n ;++i)
        low[MAXN] = INF;
    low[1] = 0;
    for(auto &x: edge[1])
        low[x.first] = x.second;
    // solve
    for(int i = 1 ;i <= n-1 ;++i){
        int low_dis = INF, low_vet = 0;
        for(int j = 1 ;j <= n ;++j)
            if(!vis[j] && low[j] < low_dis){ // travese unvis
                low_dis = low[j], low_vet = j;
            }
        vis[los_vet] = true;
        ans += low_dis;
        for(auto x: edge[low_vet]) // maintain low[]
            low[x.first] = min(low[x.first], x.second);  
	}
}
```

### 图的遍历

#### 1. 广度优先搜索 BFS

```c++
//record father and dis
bool vis[MAXN] = {0};
int father[MAXN] = {0};
int dis[MAXN] = {0};  // distance from s
vector<int> connect[MAXN];
void bfs(int s){
    queue<int> q;  // should be replaced by 			  	
    q.push(s);		// priority_queue in some cases
    father[s] = s;
    dis[s] = 0;
    while(q.size() != 0){
     	u = q.pop();
        for(auto &v: connect[u])
            if(!vis[v]){
                vis[v] = 1;
                dis[v] = dis[u] + 1;
            	father[v] = u;
                q.push(v);
            }
    }
}
```

#### 2. 深度优先搜索 DFS

```c++
// record time stamp
int time = 0;
bool vis[MAXN];
int father[MAXN];
vector<int> connect[MAXN];
int time_d[MAXN], time_f[MAXN];
void dfs(){
    for(int i = 0 ; i < MAXN ;++i)
        if(!vis[i])
            dfs_vis(i);
}
void dfs_vis(int u){
    time_d[u] = time++;
    vis[u] = 1;
    for(auto &v: connect[u])
        if(!vis[v]){
            father[v] = u;
            dfs_visit(v);
        }
    time_f[u] = time++；
}
```

### 拓扑排序

```pseudocode
TOPOLOGICAL-SORT(G){
	DFS(G) // compute v.f (finish time) for each vertex
	// add a vertex to the front of list L when it finishes
	return L
}
```

### 有向图强连通分支 

#### 1. Tarjan算法

```c++
int ts = 0， cnt = 0;
int dfn[MAXN] = {0};
int low[MAXN] = {0};
int belong[MAXN] = {0};
bool instack[MAXN] = {0};
stack<int> s;
void dfs(int pos [,int pre]) {   // 仅在无向图需要pre
    instack[pos] = true;  // instack判断当前点是否在栈内
    s.emplace(pos);  // push into stack
    low[pos] = dfn[pos] = ++ts;
    for (auto &nex : edges[pos]) {
        if (dfn[nex] == 0) {  // not visited yet
            dfs(nex [, pos]);
            low[pos] = min(low[pos], low[nex]);
        } else
            if (instack[nex] [&& nex != pre]) //无向图需要后条件
                low[pos] = min(low[pos], dfn[nex]);
    }
    if (low[pos] == dfn[pos] && instack[pos]) { // find new
        ++cnt;    // total num of strong connected components
        while (!s.empty() && s.top() != pos) {
            belong[s.top()] = cnt;
            instack[s.top()] = false;
            s.pop();
        }
        if (!s.empty()) s.pop();
        belong[pos] = cnt;
        instack[pos] = false;
    }
}
```

实际运行效果Tarjan算法优于Kosaraju算法

一些无向图的题目也会需要用Tarjan来寻找特殊的连通分量（例如任意两点间存在经过不同edge的path），在无向图上用Tarjan才需要pre参数

#### 2. Kosaraju算法

```pseudocode
// O(V+E)
STRONGLY-CONNECTED-COMPONENTS{
	DFS(G) // compute v.f
	compute G’ // invert every edge of G
	DFS(G’)  // main loop of DFS in decreasing order of v.f
	// each tree in DFS(G') is a strongly-connected-component
}
```

### 单源最短路径算法

#### 1. BELLMAN-FORD算法

```pseudocode
// O(V+E)
Bellman-Ford(G, s){
	// initialization
	for each vertex v{
		distance[v] = (v == s) ? 0 : +infinity
		predecessor[v] = NULL
	}
	// relaxation
	for i from 1 to |V|-1
		for each edge (u, v)
			if distance[u] + weight(u,v) < distance[v]{
				distance[v] = distance[u] + weight(u,v)
				predecessor[v] = u
			}
	// check negative cycle
	for each edge(u,v)
		if distance[u] + weight(u,v) < distance[v]
			error "negative cycle"
}
```

#### 2. DAG单源最短路径

```pseudocode
// O(V+E)
DAG-SHORTEST-PATH(G, s){
	topologically sort the vertices of G
	INITIALIZE-SINGLE-SOURCE(G, s)
	for each vertex u, taken in topologically sorted order
		for each vertex v in Adj[u]
			if distance[u] + weight(u,v) < distance[v]{
				distance[v] = distance[u] + weight(u,v)
				predecessor[v] = u
			}	
}
```

#### 3. Dijkstra算法

```pseudocode
// O((V+E)logV) with priority queue
int n;
bool vis[MAXN] = {0};
int dis[MAXN] = {0};
vector<pair<int, int>> edges[MAXN];
void dijkstra(G, s){ // can add predecessor if necessary
	for(int i = 1 ;i <= n ;++i)		// init
		dis[i] = INF;
	dis[s] = 0;
	priority_queue<pair<int, int>> pq;
	pq.push(0, s);
	while(!pq.empty()){
		pair<int, int> cur = pq.top();
		pq.pop();
		int u = cur.first;
		if(vis[u]) continue;
		vis[u] = true;
		for(auto &x: edges[u]){
			int v = x.second, weight = x.first;
			if(dis[u] + weight < dis[v]){
				dis[v] = dis[u] + weight;
				pq.push(dis[v], v);
			}
		}
	}
}
```

### 多源最短路径算法

#### 1. Floyd算法

```c++
int dis[MAXN][MAXN]; // store edges before floyd
    
void floyd(){
  for(int k = 1 ;k <= n ;++k)
    for(int i = 1 ;i <= n ;++i)
      for(int j = 1 ;j <= n ;++j)
        dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);
}
```

可以用Floyd算法传递闭包：

```C++
int T[MAXN][MAXN];
// (i, j)exists or i == j   =>   T[i][j] = 1
// otherwise  =>  T[i][j] = 0

void floyd(){
    for(int k = 1 ;k <= n ;++k)
        for(int i = 1 ;i <= n ;++i)
            for(int j = 1 ;j <= n ;++j)
                T[i][j] = T[i][j] || (T[i][k] && T[k][j]);
}
```

### 匹配算法

#### 1. 匈牙利算法-二部图最大匹配

```c++
//O(VE)
bool used[MAXN];
int belong[MAXN];

bool find(int x){
    for(auto& y : edges[x]){
        if(!used[y]){
            used[y] = true;
            if(belong[y]==0 || find(belong[y])){ //belong[y]!
                belong[y] = x;
                return true;
            }
        }
    }
    return false;
}

int hungarian(){
    int total = 0;
    for(int i = 1 ;i <= n ;++i){
        memset(used, 0, sizeof(used));//clear used every time
        if(find(i))
            ++total;
    }
    return total;
}
```

### 网络流

#### 1. Dinic算法(邻接矩阵)

```c++
// O(V^2E)
int edge[MAXN][MAXN];
int depth[MAXN];
bool vis[MAXN];
int src, dst;

int dfs(int u, int lim){
    if(u == dst)
        return lim;
    vis[u] = true;
    for(int v = 1 ;v <= MAXN ;++v){ // careful with range
        int ret;
        if(!vis[v] && edge[u][v]>0 && depth[v] == depth[u]+1 
           && (ret = dfs(v, min(edge[u][v], lim)))){
            edge[u][v] -= ret;
            edge[v][u] += ret;
            return ret;
        }
    }
    return 0;
}

bool bfs(){
    memset(depth, -1, sizeof(depth));
    queue<int> q;
    q.push(src);
    depth[src] = 0;
    while(!q.empty()){
        int cur = q.front();
        q.pop();
        for(int i = 1 ;i <= MAXN ;++i)
            if(edge[cur][i] > 0 && depth[i] == -1){
                depth[i] = depth[cur] + 1;
                q.push(i);
            }
    }
    return depth[dst] != -1;
}

int dinic(){
    int flow = 0;
    while(bfs()){
        int ret;
        memset(vis, 0, sizeof(vis));
        while((ret = dfs(src, INF)) != 0) // caution INF here
        	flow += ret;
    }
    return flow;
}

// add edges:
// directed single edge: edge[u][v] = w;
// directed multiple edge: edge[u][v] += w;
// undirected single edge: edge[u][v] = edge[v][u] = w;
// undirected multiple edge: edge[u][v] += w;edge[v][u] += w;

```

#### 2. Dinic算法(vector)

```c++
int src, dst;
struct node{
    int to, cap, rev;
    node(int x, int y, int z):to(x), cap(y), rev(z){};
};
vector<node> edge[MAXN];
int depth[MAXN];
bool vis[MAXN];

int dfs(int u, int lim){
    if(u == dst)
        return lim;
    vis[u] = true;
    for(auto &v: edge[u]){
        int ret;
        if(!vis[v.to] && v.cap && depth[v.to] == depth[u] + 1          && (ret = dfs(v.to, min(v.cap, lim)))){
            v.cap -= ret;
            edge[v.to][v.rev].cap += ret;
            return ret;
        }
    }
    return 0;
}

bool bfs(){
    memset(depth, -1, sizeof(depth));
    queue<int> q;
    q.push(src);
    depth[src] = 0;
    while(!q.empty()){
    	int cur = q.front();
        q.pop();
        for(auto &v: edge[cur])
        	if(depth[v.to] == -1 && v.cap > 0){
                q.push(v.to);
                depth[v.to] = depth[cur] + 1;
            }    
    }
    return depth[dst] != -1;
}

int dinic(){
    int flow = 0;
    while(bfs()){
        int ret;
        memset(vis, 0, sizeof(vis));
        while((ret = dfs(src, INF)))
            flow += ret;
    }
    return flow;
}

// add edges (assume single directed edge)
// cin >> u >> v >> w;
// edge[u].emplace_back(v, w, edge[v].size());
// edge[v].emplace_back(u, w, edge[u].size() - 1);
```

