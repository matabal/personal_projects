# Directed Graph Header File #
The directed_graph.hpp file inside this repository can be thought as a mini library. It has two classes in it, namely Vertex and DirectedGraph. As the name suggests, the aims of these classes is to create a container for organizing data in a directed graph manner. The classes cover all the fundamental functionality like cycle detection and reachability check alongside with some higher-level functionality like topological sorting, strongly connected components identification, shortest path obtaining and so forth. Thus, if you are stuck at solving one of these yourself, you may take a look into this code in order to gain an idea.    

I benefitted various widely-known (and thought) algorithms while defining many operations, and even though I acknowledged this in the comments I want to do this one more time here, a million thanks to Edsger W. Dijkstra, Robert Tarjan and many other computer science titans for inventing these brilliant algorithms and sharing them with us.   
   
Also I would like to state that this header file is a great place to start on computational graph theory or if you are dealing with relatively small to medium sized graphs. It works on decent speeds and I believe I made it robust enough. However, if you are planning to use it for large graphs (e.g. >5000 nodes) you may want to look out for faster options. Similarly, if you plan to use it for commercial purposes, or in systems that are too crucial to fail, I would encourage you to either rigorously test and improve this code or look out for more robust options.
   
**Tech Details:**   
1. The code is written for modern C++ usage (C++ 11 and above) and it is tested in MinGW and MSVC compilers. If you are intending to work with it, I personally recommend using MSVC (and Visual Studio of course). 
2. The graph is a positive weight graph, such that edges cannot have negative weights. If you try to create an edge with a negative weight, it won't be created in the version shared (of course you may customize this but some functions like shortestPath will loose functionality in such case so if you do this edit other functions to be coherent with negative weighted graphs)
3. Using the default constructor for creating a vertex will result in creation of a vertex that cannot be inserted into the graph. Similary, any vertex with an id = -1 cannot be inserted into the graph (this is just a convention for protecting the graph from empty vertices.) Hence, please make sure you at least assign an id (that isn't -1) and a weight (that isn't a negative number) while instantiating the vertices to be inserted into the graph.
4. For starting to use it, you may either clone this repository inside your c++ project (preferably inside the folder for header files) directly, or alternatively you may just copy and paste it to a header file you will create. If you go with the latter option, please make sure the header file you will create is named "directed_graph.hpp".
   
On a final note, I might be mistaken in some parts (especially time-complexities). If you have spotted a bug requires fixing, or you have great idea about an improvement please contant me from mehmetatabal3--at--gmail.com

**Disclaimer:**   
The code (directed_graph.hpp) is open-source and it can be used however the user desires. But I strongly discourage usage for violent, hateful, unethical, criminal behaivour and especially academical misconduct. That being said, I also do not accept any personal liability for outcomes of usage of this code (i.e. directed_graph.hpp). 