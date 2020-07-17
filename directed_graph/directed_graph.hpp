/*
 * directed_graph.hpp
 *
 *  Created on: Jun 28, 2020
 *  Author: Mehmet Ata Bal
 */

#ifndef DIRECTED_GRAPH_HPP_
#define DIRECTED_GRAPH_HPP_

#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <stack>
#include <unordered_map>
#include <float.h>


template <typename T>
class Vertex {

private:
	int id; //Id of the vertex
	double weight; //Weight of the vertex
	T data; //Custom data to be stored inside the vertex

public:
	Vertex() : id(-1), weight(0.0) {} //Default constructor. This will create a non-existent vertex (i.e. id = -1) which cannot be inserted to the graph
	Vertex(int x, double y, T d) : id(x), weight(y), data(d) {} //Constructor with custom data type T
	Vertex(int x, double y) : id(x), weight(y) {} //Alternative constructor without custom data type T, for graph use only
	const int getId() { return id; }
	double getWeight() { return weight; }
	T getData() { return data; }
	void setData(T data) { this->data = data; };
	void setWeight(double weight) { if (weight >= 0) { this->weight = weight; } }
};


template <typename T>
class DirectedGraph {

private:
	std::unordered_map<int, Vertex<T>*> vertices; //Stores vertices
	std::unordered_map<int, std::unordered_map<int, double>> adj_list; //Stores the graph in adjacency list format. Inner-most double type variable stores edge weight.
	size_t n_edges; //Stores total number of edges
	size_t n_vertices; //Stores total number of vertices
	int is_acyclic; //Variable to record if the graph is acyclic or not. Convention for this is following, 1: Graph is acyclic, 0: Graph is not acyclic, -1: Not tested yet

public:

	DirectedGraph();
	~DirectedGraph();

	bool contains(const int&) const; //Returns true if the graph contains the given vertex_id, false otherwise.
	bool adjacent(const int&, const int&); //Returns true if the first vertex is adjacent to the second, false otherwise.

	void addVertex(Vertex<T>&); //Adds the passed in vertex to the graph (with no edges).
	void addEdge(const int&, const int&, const double&); //Adds a weighted edge from the first vertex to the second.

	void removeVertex(const int&); //Removes the given vertex. Should also clear any incident edges.
	void removeEdge(const int&, const int&); //Removes the edge between the two vertices, if it exists.

	size_t inDegree(const int&); //Returns number of edges coming in to a vertex.
	size_t outDegree(const int&); //Returns the number of edges leaving a vertex.
	size_t degree(const int&); //Returns the degree of the vertex (both in edges and out edges).

	size_t numVertices(); //Returns the total number of vertices in the graph.
	size_t numEdges() const; //Returns the total number of edges in the graph.

	std::unordered_map<int, Vertex<T>> getVertices(); //Returns a vector containing all the vertices.
	Vertex<T>* getVertex(const int& u_id); //Retruns specified vertex. If vertex doesn't exist, the id and weight of the returned vertex are both -1. 
	double getEdgeWeight(const int& u_id, const int& v_id); //Returns the weight of the specified edge. If the edge doesn't exist, it returns -1.

	std::vector<Vertex<T>*> getNeighbours(const int&); //Returns a vector containing all the vertices reachable from the given vertex. The vertex is not considered a neighbour of itself.
	std::vector<Vertex<T>*> getSecondOrderNeighbours(const int&); // Returns a vector containing all the second_order_neighbours (i.e., neighbours of neighbours) of the given vertex.
															  // A vector cannot be considered a second_order_neighbour of itself.
	bool reachable(const int&, const int&); //Returns true if the second vertex is reachable from the first (can you follow a path of out-edges to get from the first to the second?). Returns false otherwise.
	bool containsCycles(); // Returns true if the graph contains cycles, false otherwise.

	std::vector<Vertex<T>*> depthFirstTraversal(const int&); //Returns the vertices of the graph in the order they are visited in by a depth-first traversal starting at the given vertex.
	std::vector<Vertex<T>*> breadthFirstTraversal(const int&); //Returns the vertices of the graph in the order they are visited in by a breadth-first traversal starting at the given vertex.

	/*
	 * Following function is an iterative implementation of Dijkstra's SP algorithm.
	 * It returns a pair consisting of an array of shortest distances to all other
	 * vertices from the given root vertex u_id (vertices are identified via
	 * indexes in the array such that shortest distance to vertex i is placed to
	 * the i th element in the array), and a "previous vertex" unordered_map. (If
	 * you are unsure about what a "previous vertex" list is,
	 * see https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
	 */
	std::pair<std::vector<double>, std::unordered_map<int, int>> dijkstra(int u_id);
	std::pair<double, std::vector<Vertex<T>*>> shortestPath(int u_id, int v_id); //This function finds the shortest path to a single given target vertex (v_id) from a given vertex (u_id) as a pair that contains <distance, path>

	std::vector<std::vector<Vertex<T>*>> stronglyConnectedComponents(); //Identifies and returns strongly connected components as a vector of vectors
	std::vector<Vertex<T>*> topologicalSort(); //Returns a topologically sorted list of the graph. It requires the graph to be acyclic. If the graph isn't acyclic, it returns an empty vector.


};

/*
 * Time Complexity Notes:
 * In Time Complexity statements below, the
 * symbol "n" is used to represent the number of
 * vertices, and "e" is used to represent the number of edges.
 *
 * Unless this graph is used for gigantic amounts of vertices/edges,
 * Worst-Case scenarios are not the decisive factors in terms of
 * the time complexities of the operations. Overall, time complexity
 * for worst case scenarios were calculated with multiples of n due to
 * the collision factor in STL unordered_map's hashing algorithm. However,
 * in order for this to occur in real use, there must be a collision
 * for each element in the unordered_map which is highly unlikely unless
 * data stored is considerably large. Therefore Average-Case time complexities can
 * be taken into consideration for a more realistic view and for large amounts of
 * data, time complexity will be somewhere between Average-Case and Worst-Case scenarios.
*/

template <typename T>
DirectedGraph<T>::DirectedGraph()
{
	this->n_vertices = 0;
	this->n_edges = 0;
	this->is_acyclic = -1;
	//Constructor initializes the number of vertices and edges to zero indicating graph is empty upon initialization.
}

template <typename T>
DirectedGraph<T>::~DirectedGraph() {}

template <typename T>
bool DirectedGraph<T>::contains(const int& u_id) const
{
	if (vertices.find(u_id) == vertices.end()) { return false; } //This line uses std::unordered_map's find() function to search for the given ID in the vertices list
	else { return true; }
	//Time Complexity: Average: O(1), Worst-Case: O(n)
}

template <typename T>
bool DirectedGraph<T>::adjacent(const int& u_id, const int& v_id)
{
	if (adj_list.find(u_id) != adj_list.end()) //First checks if the root vertex has any neighbours by checking if it exists in the adjacency list
	{
		if (adj_list[u_id].find(v_id) != adj_list[u_id].end()) { return true; } //Then checks if target vertex is a neighbour of the root
		return false;
	}
	return false;
	//Time Complexity: Average: O(1), Worst-Case: O(n)
}

template <typename T>
void DirectedGraph<T>::addVertex(Vertex<T>& u)
{
	if (!contains(u.getId()) && u.getId() != -1)
	{
		vertices[u.getId()] = &u;
		++n_vertices;
	}
	//Note: Added vertex is unconnected at this point. It must be connected to be initialized in adj_list.
	//Note: id = -1 is a convention for an empty vertex, thus if id is initialized as -1 or if default Vertex constructor is used for creating the vertex, it won't be added to the graph.
	//Time Complexity: Average: O(1), Worst-Case: O(n)
}

template <typename T>
void DirectedGraph<T>::addEdge(const int& u_id, const int& v_id, const double& weight)
{
	
	if (contains(u_id) && contains(v_id) && weight >= 0)
	{
		if (!adjacent(u_id, v_id)) { //Checks if given vertices are already adjacent or not
			adj_list[u_id][v_id] = weight;
			is_acyclic = -1; //This line sets the status information regarding containing cycles or not to unknown.
			++n_edges; //If they aren't adjacent, creates the edge and increments the number of vertices.
		}
		else
		{
			adj_list[u_id][v_id] = weight; //If they are adjacent, just updates the weight.
		}
	}
	//Note: If the edge weight is negative, the edge won't be created.
	//Time Complexity: Average: O(1), Worst-Case: O(n^2)
}

template <typename T>
void DirectedGraph<T>::removeVertex(const int& u_id)
{
	if (adj_list.find(u_id) != adj_list.end()) //Following if-clause disconnects the vertex to be removed by first removing all the edges directed towards it, then removing the edges it is directing towards by deleting it from the adj_list.
	{
		int s = adj_list[u_id].size(); //Variable is to keep track of the number to be subtracted from n_edges variable stored.

		for (auto &x : adj_list) //This loop iterates through all of the adjacency list and when it finds an edge towards the vertex to be removed, it deletes that edge.
		{
			if (x.second.find(u_id) != x.second.end())
			{
				x.second.erase(u_id);
				++s;
			}
		}
		adj_list.erase(u_id); //Removes the vertex from the adjacency list.
		n_edges -= s;
	}

	if (contains(u_id)) //Following if-clause removes the vertex itself if it exists.
	{
		vertices.erase(u_id);
		--n_vertices;
	}
	//Time Complexity: Average: O(n), Worst-Case: O(n^2)
}

template <typename T>
void DirectedGraph<T>::removeEdge(const int& u_id, const int& v_id)
{
	if (adjacent(u_id, v_id)) //Following if-clause deletes the given edge (from u_id, towards v_id) if it exists
	{
		adj_list[u_id].erase(v_id);
		--n_edges;
	}
	//Time Complexity: Average: O(1), Worst-Case: O(n)
}

template <typename T>
size_t DirectedGraph<T>::inDegree(const int& u_id)
{
	size_t s = 0;
	for (auto &x : adj_list)
	{
		if (x.second.find(u_id) != x.second.end()) { s++; } //Increments the variable whenever it finds an edge pointing to the given vertex
	}
	return s;
	//Time Complexity: Average: O(n), Worst-Case: O(n^2)
}

template <typename T>
size_t DirectedGraph<T>::outDegree(const int& u_id)
{
	return adj_list[u_id].size();
	//Time Complexity: Average: O(1), Worst-Case: O(n)
}

template <typename T>
size_t DirectedGraph<T>::degree(const int& u_id)
{
	return inDegree(u_id) + outDegree(u_id);
	//Time Complexity: Average: O(n), Worst-Case: O(n^2)
}

template <typename T>
size_t DirectedGraph<T>::numVertices()
{
	return n_vertices;
	//Time Complexity: Worst-Case: O(1)
}

template <typename T>
size_t DirectedGraph<T>::numEdges() const
{
	return n_edges;
	//Time Complexity: Worst-Case: O(1)
}

template <typename T>
std::unordered_map<int, Vertex<T>> DirectedGraph<T>::getVertices()
{
	return vertices;
	//Time complexity: Worst-Case: O(1)
}

template <typename T>
Vertex<T>* DirectedGraph<T>::getVertex(const int& u_id)
{
	if (contains(u_id))
	{
		return &vertices[u_id];
	}

	return &Vertex<T>(-1, -1.0); //If vertex doesn't exist, the function will return a vertex with id and weight -1.
	//Time Complexity: Average: O(1) Worst-Case: O(n)
}

template <typename T>
double DirectedGraph<T>::getEdgeWeight(const int& u_id, const int& v_id)
{
	if (adj_list.find(u_id) != adj_list.end())
	{
		if (adj_list[u_id].find(v_id) != adj_list[u_id].end()) { return adj_list[u_id][v_id]; }
		return -1.0;
	}
	return -1.0;
	//Time Complexity: Average: O(1), Worst-Case: O(n^2)
}


template <typename T>
std::vector<Vertex<T>*> DirectedGraph<T>::getNeighbours(const int& u_id)
{
	std::vector<Vertex<T>*> neighbours;
	if (!adj_list[u_id].empty()) //Checks if given vertex has any neighbours
	{
		for (auto &x : adj_list[u_id]) { neighbours.push_back(vertices[x.first]); }
	}//Similar to get_vertices, iterates through all the adjacent vertices and saves them in a vector
	return neighbours;
	//Time complexity: Average: O(n), Worst-Case: O(n^2)
}

template <typename T>
std::vector<Vertex<T>*> DirectedGraph<T>::getSecondOrderNeighbours(const int& u_id)
{
	std::vector<Vertex<T>*> second_neighbours;
	std::unordered_map<int, int> visited; //Map of visited vertices

	if (!adj_list[u_id].empty())
	{
		for (auto& x : adj_list[u_id]) //This loop iterates over each neighbour of vertex of interest
		{
			for (auto& y : adj_list[x.first]) //This loop iterates over neighbours of each neighbour vertex identified in outer loop
			{
				if (visited.find(y.first) == visited.end() && y.first != u_id) //Checks if second order vertex identified already exists in the list and also if second order found is vertex of interest or not
				{
					second_neighbours.push_back(vertices[y.first]);
					visited[y.first] = y.first;
				}

			}
		}
	}
	return second_neighbours;
	//Time Complexity: Average: O(Avg. Number of Neighbours per Vertex^2), Worst Case: O(n^4)
}

template <typename T>
bool DirectedGraph<T>::reachable(const int& u_id, const int& v_id)
{
	//This function is a Depth First Search Algorithm that halts when latter vertex is found
	//Returns true if v_id is reachable from u_id

	std::stack<int> track; //Stack for DFS
	std::vector<bool>  visited (numVertices(), false);

	track.push(u_id); //Initialize the algorithm by pushing the root into the stack
	while (!track.empty()) //Stack being empty indicates all reachable graph is traversed hence loop until it is empty, hence this while loop will run until this condition is met
	{
		int curr_id = track.top();
		bool found = false;
		auto it = adj_list[curr_id].begin();
		while (it != adj_list[curr_id].end() && !found) //
		{	
			if (!visited.at(it->first)) //Checks if current vertex is already visited
			{
				if (it->first == v_id) { return true; }  //This if-clause returns true if current vertex is vertex to be found hence indicating it is reachable from given former vertex
				visited.at(it->first) = true;
				track.push(it->first); //Push current vertex into into stack so we can iterate over it's neighbours
				found = true; //Algorithm found a unvisited neighbour, hence, the inner-loop needs to halt. (see DFS).
			}
			++it;
		}
		if (!found) { track.pop(); } //If no unvisited neighbour is found, we are done with that vertex hence pop it out of stack.
	}
	return false;
	//Time Complexity: Average: Hard to determine, it is based on how close the latter vertex or adj_list storage order, Worst-Case: O(n)
}

template <typename T>
bool DirectedGraph<T>::containsCycles()
{
	//If information regarding beign cyclic already obtained previously, it returns this information. 
	//Note that, after each edge addition, the status for this information is changed to unknown (see: addEdge function).
	if (is_acyclic == 1) { return false; }
	else if (is_acyclic == 0) { return true; }

	for (auto& x : adj_list)//Following for-each loop checks each vertex in adj_list is reachable from itself. If a vertex is reachable from itself, it means the graph is cyclic (contains cycles).
	{
		if (reachable(x.first, x.first))
		{
			is_acyclic = 0;
			return true;
		}
	}
	is_acyclic = 1;
	return false;
	//Time Complexity: Average: Hard to tell, depends on if cycles exist or their location in adj_list, Worst-Case: if n > e; O(n^2), if e > n; O(e*n)
}

template <typename T>
std::vector<Vertex<T>*> DirectedGraph<T>::depthFirstTraversal(const int& u_id)
{
	//This is an iterative implementation of a Depth First Traversal algorithm (as opposing to Search). Thus it is very similar to the DFS defined in the reachable() function.
	//Hence not all the lines will be commented, only changed ones will be.
	std::vector<Vertex<T>*> ordered; //This is the list to push all the vertices visited and to be returned at the end
	std::stack<int> track;
	std::vector<bool>  visited (numVertices(), false);

	track.push(u_id); //Initialize the algorithm by pushing the root into the stack
	ordered.push_back(vertices[u_id]); //And visiting it
	visited.at(u_id) = true;
	while (!track.empty())
	{
		int curr_id = track.top();
		bool found = false;
		auto it = adj_list[curr_id].begin();
		while (it != adj_list[curr_id].end() && !found)
		{
			if (!visited.at(it->first))
			{
				ordered.push_back(vertices[it->first]); // In this line, unlike DFS, rather than returning and killing the function, the function adds the unvisited vertex to the ordered list
				visited.at(it->first) = true;
				track.push(it->first);
				found = true;
			}
			++it;
		}
		if (!found) { track.pop(); }
	}
	return ordered;
	//Time-Complexity: Average: Hard to tell, depends on the root vertex given, Worst-Case O(n+e)
}

template <typename T>
std::vector<Vertex<T>*> DirectedGraph<T>::breadthFirstTraversal(const int& u_id)
{
	/*
	 * This is an iterative implementation of a Breadth-First Traversal algorithm. It is pretty analogous to Depth-First Traversal defined above.
	 * It differs from the DFT above in two points. It uses a queue instead of a stack and it doesn't break the inner loop when a new unvisited vertex
	 * is found. Rather, it pushes all the unvisited neighbours to the queue first and then moves on the the one at the front of the queue.
	 */
	std::vector<Vertex<T>*> ordered; //List to populate in BFT and return
	std::queue<int> track;
	std::vector<bool>  visited (numVertices(), false);
	
	ordered.push_back(vertices[u_id]); //Initialize the algorithm by visiting the given root vertex,
	track.push(u_id); //pushing it into the queue
	visited.at(u_id) = true; //and marking it as visited.
	while (!track.empty())//If the queue is empty, the function visited all reachable graph hence it must halt.
	{
		auto it = adj_list[track.front()].begin();
		while (it != adj_list[track.front()].end()) //As opposing to the DFT, we are iterating over each neighbour of the vertex at front in this while loop
		{
			if (!visited.at(it->first))
			{
				ordered.push_back(vertices[it->first]);
				track.push(it->first);
				visited.at(it->first) = true;
			}
			++it;
		}
		track.pop(); //Pop the current vertex when all of its neighbours are visited
	}
	return ordered;
	//Time-Complexity: Average: Hard to tell, depends on the root vertex given, Worst-Case O(n+e)
}

template <typename T>
std::pair<std::vector<double>, std::unordered_map<int, int>> DirectedGraph<T>::dijkstra(int u_id)
{
	//This function is an iterative implementation of Dijkstra's Shortest Path Algorithm
	std::vector<double> dist (numVertices(), DBL_MAX); //This array is to store the shortest path founded to other vertices from root. It will get dynamically updated. It is initialized as all values being equal to inifinity (i.e. INT_MAX: the practical limit) since distances are unknown.
	dist.at(u_id) = 0; //However distance to root (itself) is known initially and it is zero hence initialize it as such.

	typedef std::pair<double, int> pi; //This line defines an stl pair type later to be use with the min heap. This pair consists of <shortest distance from root, vertex id>
	std::priority_queue<pi, std::vector<pi>, std::greater<pi>> min_heap; //This is a min heap data structure defined using stl priority_queue. It accepts pairs as nodes and first element in a pair is used for comparison in min heap.
	min_heap.push(std::make_pair(dist.at(u_id), u_id)); //Push the root as the first element (since dist. to root is 0 and no negative edge weights exists, root will always stay heap root)

	std::vector<bool> visited (numVertices(), false); //An auxiliary vector to flag visited vertices
	std::unordered_map<int, int> previous_vertex; //this is an stl unordered map that will store the source vertex (aka. previous vertex) of any vertex in the dist array

	while (!min_heap.empty()) //loop until min heap is empty which will mark all vertices are visited
	{
		auto current = min_heap.top(); // each iteration get the vertex at the top of min heap
		auto curr_neighbours = getNeighbours(current.second); // get current vertex's neighbours
		
		for (auto& x : curr_neighbours) // loop in current vertex's neighbours
		{
			auto edge_weight = getEdgeWeight(current.second, x->getId()); //get the weight of edge between current vertex and current neighbour x     

			if (!visited.at(x->getId()) && (dist.at(current.second) + edge_weight) <= dist.at(x->getId())) // if current neighbour is not visited and found distance (i.e. distance via the current vertex) to the current neighbour (from root) is less than the recorded distance (i.e. the one in the dist array) 
			{
				dist.at(x->getId()) = dist.at(current.second) + edge_weight; // update the shortest distance to current neighbour (from root) as new distance obtained (via current vertex).
				previous_vertex[x->getId()] = current.second; // update the source vertex of the current neighbour as the current vertex
				min_heap.push(std::make_pair(dist.at(x->getId()), x->getId())); // push current neighbour to the min heap
			}
			
		}
		
		visited.at(current.second) = true; // After all the neighbours are evaluated, the current node is visited hence mark it as visited
		min_heap.pop(); // and remove it from the min heap
	}
	return std::make_pair(dist, previous_vertex);
	//Time complexity: Average Case: O((v+e)logv) Worst Case: O((v+e)logv)
}

template <typename T>
std::pair<double, std::vector<Vertex<T>*>> DirectedGraph<T>::shortestPath(int u_id, int v_id)
{
	auto dijkstra_pair = dijkstra(u_id);
	std::vector<Vertex<T>*> ordered_shortest; //vector type list to store ordered vertices
	auto previous_vertex = dijkstra_pair.second;


	int curr = v_id; // take the target as current since it will backtrack
	bool reached = false;
	while (!reached)
	{
		if (curr == u_id) { reached = true; } //break the loop after this iteration if root vertex is reached
		ordered_shortest.insert(ordered_shortest.begin(), vertices[curr]); // insert current node as the first node in the list (since the path is inserted in reverse order)
		curr = previous_vertex[curr]; //change current node with its previous node in the path to root
	}

	return std::make_pair(dijkstra_pair.first.at(v_id), ordered_shortest);
	//Time complexity: Average Case: O((v+e)logv) Worst Case: O((v+e)logv)
}

template <typename T>
std::vector<std::vector<Vertex<T>*>> DirectedGraph<T>::stronglyConnectedComponents()
{
	//Following function is an iterative implementation of Tarjan's Strongly Connected Components Algorithm
	std::vector<std::vector<Vertex<T>*>> sc_components; //This is the embedded vector the program will store strongly connected components founds as separate vectors

	std::vector<int> low_link (numVertices()); //A vector to store low link values of vertices
	std::vector<int> link (numVertices()); //A vector to store link values (also referred as the id) of vertices
	std::vector<bool> visited (numVertices(), false); //A vector array to flag visited vertices as visited
	std::vector<bool> in_stack (numVertices(), false); //A vector to indicate if the vertex is currently in the tarjan stack
	int link_i = 0; //This variable will be used as a counter to assign a link value to visited vertices. It will be incremented after every assignation

	for (auto& x : adj_list) // Iterate through all the connected vertices in the graph
	{
		if (!visited.at(x.first)) //If the current vertex is not visited
		{
			std::stack<int> track; //Stack for depth first traversal
			std::stack<int> tarjan_stack; //Stack for storing current strongly connected component (i.e. tarjan stack)

			track.push(x.first); //This line and next 6 lines are to initiate the depth first traversal. Hence start by pushing current vertex into dft stack
			tarjan_stack.push(x.first); //Push the current vertex to tarjan stack
			visited.at(x.first) = true; //Mark the current vertex as visited
			in_stack.at(x.first) = true; //Mark  the current vertex as in tarjan stack
			link.at(x.first) = link_i; //Assign link value
			low_link.at(x.first) = link_i; //Assign the low link value which currently is same with the link value
			link_i++; //Increment the link value counter

			while (!track.empty()) //Loop until dft stack is empty. This line is where the dft starts
			{
				int curr_id = track.top();
				bool found = false; //This variable will be used to break the inner loop as soon as the functions finda an unvisited adjacent vertex
				auto it = adj_list[curr_id].begin();

				while (it != adj_list[curr_id].end() && !found)
				{
					if (!visited.at(it->first) && !in_stack.at(it->first)) //If the current neighbour is not visited and not in the tarjan stack
					{
						track.push(it->first); //This line and next 6 lines are the same procedure with 7 lines defined after tarjan stack's definition above
						tarjan_stack.push(it->first);
						visited.at(it->first) = true;
						in_stack.at(it->first) = true;
						low_link.at(it->first) = link_i;
						link.at(it->first) = link_i;
						link_i++;
						found = true; //This line will break the loop before the next iteration thus allow for dft.
					}
					else if (in_stack.at(it->first)) //If however current neighbour is in tarjan stack, it indicates we are back tracking on a strongly connected component
					{
						low_link.at(curr_id) = std::min(low_link.at(curr_id), low_link.at(it->first)); //Hence, do the link minimizing operation in this line by taking minimum low_link value between current neighbour and current vertex
					}
					++it;
				}
				if (!found) //If no new unvisited vertex is found it means we have to start dft backtrack
				{
					if (low_link.at(curr_id) == link.at(curr_id)) //This line checks if low link value and link value is the same whilst dft backtracking.
					{ //It is an extremely crucial line to this algorithm because a link value and a low link value being the same whilst backtrack indicates the tarjan stack currently stores one complete strongly connected component and the top element of the stack is the beginning of it (see Tarjan's algorithm)
						std::vector<Vertex<T>*> sc_component; //Hence we create a vector in order to store this newly found strongly connected component
						bool reached = false; //This is bool variable to break the loop when required amount of vertices are popped from the tarjan stack
						while (!reached) //until required amount of vertices are pooped
						{
							int tarjan_top = tarjan_stack.top();
							if (tarjan_top == curr_id) { reached = true; } //This line is to check if currently we are at the last element of the strongly connected component. If so mark reached to be true hence the loop will iterate once more and it will stop.
							sc_component.insert(sc_component.begin(), vertices[tarjan_top]); //Insert the top of tarjan stack at each iteration
							in_stack.at(tarjan_top) = false; //Mark the top to be not in the tarjan stack (it will be popped out of it in next line)
							tarjan_stack.pop(); //pop a vertex
							if (tarjan_stack.empty()) { reached = true; } //This line is to protect the code from Segmentation Fault. If the tarjan stack is empty, the loop will break
						}
						sc_components.push_back(sc_component); //Lastly, insert the identified strongly connected component list into the SCCs list defined above.
					}
					track.pop(); //Pop a vertex from dft stack to backtrack
				}
			}
		}
	}
	return sc_components;
	//Time complexity: Average: O(e+v), Worts-Case: O(e+v)
}

template <typename T>
std::vector<Vertex<T>*> DirectedGraph<T>::topologicalSort() {
	//The following function is an iterative topological sort algorithm relying on a depth first traversal logic. Note: It requires a DAG to properly function.

	std::vector<Vertex<T>*> top_ordered; //Vector to store topologically sorted vertices

	if (is_acyclic == -1) { containsCycles(); } //If the graph is still not checked for cycles, check it.

	if (is_acyclic == 1)
	{

		std::vector<bool> visited(numVertices(), false); //Array to flag visited vertices
		std::vector<bool> to_be_visited(numVertices(), false); //Array to flag vertices need visiting

		for (auto& x : adj_list)
		{
			if (!visited.at(x.first)) //If the current vertex is not vertex
			{
				std::stack<int> track; //define the dft stack
				track.push(x.first); //push the initiation vertex in the dft stack
				to_be_visited.at(x.first) = true; //mark it as to be visited

				while (!track.empty()) //loop until dft stack is empty (i.e. begin dft)
				{
					int curr_id = track.top();
					bool found = false; //This variable will be used to break the inner loop as soon as we find an unvisited adjacent vertex
					auto it = adj_list[curr_id].begin(); //Iterator to iterate over adjacent vertices of the vertex at the top of the dft stack
					while (it != adj_list[curr_id].end() && !found)  //loop over all neighbours and until next vertex (and unvisited neighbour) is not found
					{
						if (!to_be_visited.at(it->first)) //If current vertex is not marked as to be visited
						{
							track.push(it->first); //Mark it as to be visited by pushing into the dft stack
							to_be_visited.at(it->first) = true; //And flag it as such
							found = true; //Next vertex is found, break the loop before next iteration
						}
						++it;
					}

					if (!found) //If no unvisited neighbour is found
					{
						if (!visited.at(curr_id)) // Within this statement (and relatedly above while traversing adjacent vertices) this algorithm differentiates from a standard dft
						{ //Rather than visiting the vertices as it goes, by this statement, the function visits a vertex only if no unvisited neighbour vertices are found
							top_ordered.insert(top_ordered.begin(), vertices[curr_id]);
							visited.at(curr_id) = true; //And flag it as visited
						}
						track.pop(); //Backtrack by poping the last vertex
					}
				}
			}
		}
	}
	return top_ordered; //Note: The function will return an empty vector if the graph is cyclic.
	//Time-Complexity: Average: Hard to tell, depends on the root vertex given, Worst-Case O(n+e)
}

#endif /* DIRECTED_GRAPH_HPP_ */
