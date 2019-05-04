/* CSC 2/458: Parallel and Distributed Systems
 * Spring 2019
 * Final Project: Subgraph Isomorphism on GPUs (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.711.4777&rep=rep1&type=pdf)
 * Author: Soubhik Ghosh (netId: sghosh13)
 */

#include <bits/stdc++.h> 
#include <unistd.h>

//#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

std::string data_graph_file = "";
std::string query_graph_file = "";

std::vector<int> visit_order;

#define BLOCK_SIZE1D 256
#define BLOCK_SIZE2D 16

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

struct hash_table {
        int count_u;
        int *d_vertices_u;
        int *d_addresses;
        int count_v;
        int *d_vertices_v;
};

class Graph
{
private:
	using edge_list = std::vector<int>;

	int m_num_vertices = 0;
	int m_num_edges = 0;
	
	/* Host */
	std::vector<char> h_labels;
	edge_list *h_adjacency_list;

	/* Device graph */
	int *d_vertices;
	char *d_labels;
	int *d_edges;
	
public:
	Graph(int num_vertices = 1) : m_num_vertices(num_vertices) {
		h_adjacency_list = new edge_list[num_vertices];
	}

	~Graph() {
                h_labels.resize(0);
                delete [] h_adjacency_list;
                cudaFree(d_vertices);
                cudaFree(d_labels);
                cudaFree(d_edges);
	}

	void reset(int num_vertices) {
		m_num_vertices = num_vertices;
		h_labels.resize(0);
		delete [] h_adjacency_list;
		h_adjacency_list = new edge_list[num_vertices];
		
		cudaFree(d_vertices);
		cudaFree(d_labels);
		cudaFree(d_edges);
	}

	void add_label(char l) {
		h_labels.emplace_back(l);
	}

	void load(std::string &graph_file){
		int num_vertices, u, v;
		char l;
		if (graph_file.compare("-") == 0) {
			/* Read number of Drinking Philosophers */
			std::cout << "Enter number of vertices: ";
			std::cin >> num_vertices;
			this->reset(num_vertices);

			std::cout << "Enter labels: ";
			for (int i = 0;i < num_vertices;i++) {
				std::cin >> l;
				this->add_label(l);
			}

			std::cout << "Enter edges: ";
			/* Exit when pressing ctrl-d */
			while (std::cin >> u >> v)
				this->add_edge(u - 1, v - 1);
		}
		else {
			std::ifstream inf(graph_file);
			if (inf.is_open()) {
				/* Read number of Drinking Philosophers */
				inf >> num_vertices;
				this->reset(num_vertices);
				
				for (int i = 0;i < num_vertices;i++) {
					inf >> l;
					this->add_label(l);
				}

				while (inf >> u >> v)
					this->add_edge(u - 1, v - 1);
				inf.close();
			}
			else {
				throw std::string("Unable to open the config file: " + graph_file + ".\nExiting...");
			}
		}

	}

	void add_edge(int u, int v) {
		/* Test for bounded ids */
		if (u < 0 || u >= m_num_vertices || v < 0 || v >= m_num_vertices) 
			throw std::string("Sorry, these don't seem to be the correct vertices.\nExiting...");

		/* Test for self loops */
		if (u == v)
			throw std::string("Sorry no self loops allowed.\nExiting...");
		
		/*
		auto it = std::find_if(m_fork_list, m_fork_list + m_num_edges, [u, v](const Fork& f) {
			return (f.direction == std::make_pair(u, v)) || (f.direction == std::make_pair(v, u));
		});

		if (it != m_fork_list + m_num_edges)
			throw std::string("Sorry, the neighbours are allowed to share atmost one fork. \nExiting...");
		*/

		h_adjacency_list[u].emplace_back(v);
		h_adjacency_list[v].emplace_back(u);
		m_num_edges++;
	}

	void load_device() {
		cudaMalloc(&d_labels, m_num_vertices * sizeof(char));
		cudaMemcpy(d_labels, h_labels.data(), m_num_vertices * sizeof(char), cudaMemcpyHostToDevice);

		cudaMalloc(&d_vertices, (m_num_vertices + 1) * sizeof(int));	
		cudaMalloc(&d_edges, (m_num_edges << 1) + sizeof(int));

		int running_start = 0;
		for(int i = 0;i < m_num_vertices;i++) {
			cudaMemcpyToSymbol(d_vertices + i, &running_start, sizeof(int));
			cudaMemcpy(d_edges + running_start, h_adjacency_list[i].data(), h_adjacency_list[i].size() * sizeof(int), cudaMemcpyHostToDevice);
			running_start += h_adjacency_list[i].size();
		}
		cudaMemcpyToSymbol(d_vertices + m_num_vertices, &running_start, sizeof(int));
	}

	bool is_connected() const {
		/* Note: use this function only after adding all the edges */
		return m_num_edges >= (m_num_vertices - 1) && m_num_edges <= m_num_vertices * (m_num_vertices - 1) / 2;
	}

	void print() const {
		std::cout << "Number of vertices: " << m_num_vertices << std::endl;
		for (int id = 0; id < m_num_vertices; id++) {
			std::cout << "Vertex value: " << id + 1 << std::endl;
			
			std::cout << "Label: " << h_labels[id] << std::endl;

			std::cout << "Neighbors: ";
			for (auto const e : h_adjacency_list[id]) {
				std::cout << e + 1 << " ";
			}
			std::cout << "\n\n" << std::flush;
		}
	}
	
	friend void init_spanning_tree(Graph &query_spanning_tree, Graph &query_graph);
	friend bool* initialize_candidate_vertices(Graph &T, Graph &g);
	friend bool* refine_candidate_vertices(bool* c_set, Graph &q, Graph &g);
	friend std::map<std::pair<int, int>, hash_table> find_candidate_edges(bool* c_set, Graph &q, Graph &g);
};

void init_spanning_tree(Graph &query_spanning_tree, Graph &query_graph) {
	query_spanning_tree.reset(query_graph.m_num_vertices);
	/* Create the minimum spanning tree */
	query_spanning_tree.reset(6);
                        query_spanning_tree.add_label('B');
                        query_spanning_tree.add_label('A');
                        query_spanning_tree.add_label('B');
                        query_spanning_tree.add_label('A');
                        query_spanning_tree.add_label('C');
                        query_spanning_tree.add_label('C');

                        query_spanning_tree.add_edge(0, 1);
                        query_spanning_tree.add_edge(1, 4);
                        query_spanning_tree.add_edge(2, 4);
                        query_spanning_tree.add_edge(3, 4);
                        query_spanning_tree.add_edge(4, 5);
	query_spanning_tree.load_device();
	visit_order.emplace_back(4);
	visit_order.emplace_back(1);
}

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__device__ size_t d_pitch;
size_t h_pitch;

/* Checks whether b is candidate of a */
__forceinline__ __device__ bool check_candidate (char *labels1, int *vertices1, char *labels2, int *vertices2, int a, int b) {
	return labels1[a] == labels2[b] && (vertices1[a+1] - vertices1[a] <= vertices2[b+1] - vertices2[b]);
}

__global__
void kernel_check(bool *c_set, int width, int height, int u,
			int *vertices1, char *labels1, int *edges1,
			int *vertices2, char *labels2, int *edges2) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (tid >= width) return;
	
	bool* row_u = (bool*)((char*)c_set + u * d_pitch);

	if (check_candidate(labels1, vertices1, labels2, vertices2, u, tid))
		row_u[tid] = true;
}

__global__
void kernel_explore(int *c_array, int cand_count, bool *c_set, int u,
                        int *vertices1, char *labels1, int *edges1,
                        int *vertices2, char *labels2, int *edges2) {
	int warp_id = threadIdx.x / WARP_SIZE;
        int wid = warp_id + blockIdx.x * (blockDim.x / WARP_SIZE);	
	
	if (wid >= cand_count)
                return;

	int u_ = c_array[wid];

	bool* row_u = (bool*)((char*)c_set + u * d_pitch);

	for(int i = vertices1[u];i < vertices1[u+1];i++) {
		int v = edges1[i];
		bool exists = true;
		for(int j = vertices2[u_];j < vertices2[u_+1];j++) {
			int v_ = edges2[j];
			if (check_candidate(labels1, vertices1, labels2, vertices2, v, v_)) {
				exists = false;
				break;
			}
		}
		if (exists) {
			//size_t rank = __popc(FULL_MASK & __lanemask_lt());
			if (WARP_SIZE % threadIdx.x == 0) row_u[u_] = false;
			return;
		}
	}		

	int laneid = WARP_SIZE % threadIdx.x;

	if (vertices2[u_ + 1] - vertices2[u_] <= laneid) return;

        /* Its assumend that the degree is at most 32 which is the warp size */
        int v_ = edges2[vertices2[u_] + laneid];

	for(int i = vertices1[u];i < vertices1[u+1];i++) {	
		int v = edges1[i];
		
		if (check_candidate(labels1, vertices1, labels2, vertices2, v, v_)) {
			bool* row_v = (bool*)((char*)c_set + v * d_pitch);
			row_v[v_] = true;
		}
	}
}

/* Could be done with thrust API, check later */
__global__
void __kernel_collect(int *prefix_sum, int *c_array, bool *c_set, int width, int u) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= width)
                return;
        if (prefix_sum[tid + 1] > prefix_sum[tid]) {
                c_array[prefix_sum[tid]] = tid;
        }
}

__host__
int* kernel_collect(bool *c_set, int width, int u, int *cand_count) {

	int blocks_grid = (width + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;

	thrust::device_ptr<int> prefix_sum = thrust::device_malloc<int>(2);
	prefix_sum[0] = prefix_sum[1] = 0;
	
	thrust::device_ptr<bool> dev_ptr((bool*)((char*)c_set + u * h_pitch));
       	
	/* Prefix scan */
        thrust::inclusive_scan(thrust::device, prefix_sum, prefix_sum + 2, prefix_sum);
	
       	*cand_count = prefix_sum[width];
	
	int *c_array;
        cudaMalloc(&c_array, *cand_count * sizeof(int));

	__kernel_collect<<<blocks_grid, BLOCK_SIZE1D>>>(thrust::raw_pointer_cast(prefix_sum), c_array, c_set, width, u);

	thrust::device_free(prefix_sum);

	return NULL;
}

__host__ 
bool* initialize_candidate_vertices(Graph &T, Graph &g) {
	bool *c_set, *initialized;
	
	cudaMallocPitch(&c_set, &h_pitch, g.m_num_vertices * sizeof(bool), T.m_num_vertices);
	cudaMemset2D(&c_set, h_pitch, false, g.m_num_vertices * sizeof(bool), T.m_num_vertices);		
	
	cudaMemcpyToSymbol(&d_pitch, &h_pitch, sizeof(size_t));

	initialized = new bool[T.m_num_vertices]{ false };

	int blocks_grid = (g.m_num_vertices + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;

	for(auto const u:visit_order) {
		if(initialized[u] == false) {
			/* Pass query graph or spanning tree? */
			kernel_check<<<blocks_grid, BLOCK_SIZE1D>>>(c_set, g.m_num_vertices, T.m_num_vertices, u,
					T.d_vertices, T.d_labels, T.d_edges,
					g.d_vertices, g.d_labels, g.d_edges);
			cudaDeviceSynchronize();
			initialized[u] = true;
		}
	
		int cand_count;

		int *c_array = kernel_collect(c_set, g.m_num_vertices, u, &cand_count);

		int warp_blocks_grid = (cand_count * WARP_SIZE + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;

		kernel_explore<<<warp_blocks_grid, BLOCK_SIZE1D>>>(c_array, cand_count, c_set, u,
                               T.d_vertices, T.d_labels, T.d_edges,
                               g.d_vertices, g.d_labels, g.d_edges);
		
		cudaFree(c_array);

		for(auto const v:T.h_adjacency_list[u]) {
			initialized[u] = true;	
		}	
	}

	delete initialized;
	return c_set;
}

__global__
void kernel_explore_prune(int *c_array, int cand_count, volatile bool *c_set, int u,
                        int *vertices1, char *labels1, int *edges1,
                        int *vertices2, char *labels2, int *edges2, bool *continue_refinement) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid >= cand_count) return;

        int u_ = c_array[tid];

        bool* row_u = (bool*)((char*)c_set + u * d_pitch);
	
	for(int i = vertices1[u];i < vertices1[u+1];i++) {
		int v = edges1[i];
		/* Ignoring low connectivity vertices */
		if (vertices1[v+1] - vertices1[v] > 1) {
			bool exists = true;
			for(int j = vertices2[u_];j < vertices2[u_+1];j++) {
				int v_ = edges2[j];
				bool* row_v = (bool*)((char*)c_set + v * d_pitch);
				/* Using the c_set array generated in the initialization step and reduce random accesses */
				if (row_v[v_] == true) {
					exists = false;
					break;
				}
			}
			if (exists) {
				*continue_refinement = true;
				row_u[u_] = false;
				return;
			}
		}
	}
}

__host__
bool* refine_candidate_vertices(bool *c_set, Graph &q, Graph &g) {
	bool *d_cr, h_cr = false;
	
	cudaMalloc(&d_cr, sizeof(bool));	
	
	cudaMemcpyToSymbol(d_cr, &h_cr, sizeof(bool));

	/* Use visit order? */
	for(int u = 0;u < q.m_num_vertices;u++)	{
		/* Ignoring low connectivity vertices */
		if (q.h_adjacency_list[u].size() > 1) {
                
			int cand_count;

                	int *c_array = kernel_collect(c_set, g.m_num_vertices, u, &cand_count);

                	int warp_blocks_grid = (cand_count + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;

                	kernel_explore_prune<<<warp_blocks_grid, BLOCK_SIZE1D>>>(c_array, cand_count, c_set, u,
                               q.d_vertices, q.d_labels, q.d_edges,
                               g.d_vertices, g.d_labels, g.d_edges, d_cr);

			cudaFree(c_array);
		}
	}

	cudaMemcpyFromSymbol(&h_cr, d_cr, sizeof(bool));
	cudaFree(d_cr);
	if (h_cr) {
		c_set = refine_candidate_vertices(c_set, q, g);
	}
	return c_set;
}

__global__
void kernel_count_candidate_edges(int *d_vertices_u, int *d_addresses, int count_u, bool *c_set, int u, int v,
                        int *vertices1, char *labels1, int *edges1,
                        int *vertices2, char *labels2, int *edges2) {
        int warp_id = threadIdx.x / WARP_SIZE;
        int wid = warp_id + blockIdx.x * (blockIdx.x / WARP_SIZE);

        if (wid >= count_u)
                return;

        int u_ = d_vertices_u[wid];

	bool* row_v = (bool*)((char*)c_set + v * d_pitch);

	int laneid = WARP_SIZE % threadIdx.x;

	if (vertices2[u_ + 1] - vertices2[u_] <= laneid) return;

	/* Its assumend that the degree is at most 32 which is the warp size */
	int v_ = edges2[vertices2[u_] + laneid];

	atomicAdd(&d_addresses[wid + 1], row_v[v_]);
}


__global__
void kernel_fill_candidate_edges(int *d_vertices_u, int *d_addresses, int *d_vertices_v, int count_u, bool *c_set, int u, int v,
                        int *vertices1, char *labels1, int *edges1,
                        int *vertices2, char *labels2, int *edges2) {
        int warp_id = threadIdx.x / WARP_SIZE;
        int wid = warp_id + blockIdx.x * (blockIdx.x / WARP_SIZE);

        __shared__ int temp[BLOCK_SIZE1D / WARP_SIZE];
	
	if (wid >= count_u) return;

        int u_ = d_vertices_u[wid];

	bool* row_v = (bool*)((char*)c_set + v * d_pitch);

        int laneid = WARP_SIZE % threadIdx.x;

	if (vertices2[u_ + 1] - vertices2[u_] <= laneid) return;

        /* Its assumed that the degree is at most 32 which is the warp size */
        int v_ = edges2[vertices2[u_] + laneid];
	
	temp[wid] = 0; __syncwarp();
	
	int i = atomicAdd(&temp[wid], row_v[v_]);

	if(row_v[v_] == true)
		d_vertices_v[d_addresses[wid] + i] = v_;	
}


__host__
std::map<std::pair<int, int>, hash_table> find_candidate_edges(bool *c_set, Graph &q, Graph &g) {

 	std::map<std::pair<int, int>, hash_table> E;

       	hash_table ht;	
	/* Read distinct edges */
	for(int u = 0;u < q.m_num_vertices;u++) {
                for(auto const v:q.h_adjacency_list[u]) {
	
			/* Check if we already visited the edge*/
			if (E.find(std::make_pair(u, v)) != E.end() || E.find(std::make_pair(v, u)) != E.end()) {
				continue;
			}

                	ht.d_vertices_u = kernel_collect(c_set, g.m_num_vertices, u, &ht.count_u);

                        cudaMalloc(&ht.d_addresses, (ht.count_u + 1) * sizeof(int));

                        cudaMemset(ht.d_addresses, 0, (ht.count_u + 1) * sizeof(int));
				
			int warp_blocks_grid = (ht.count_u * WARP_SIZE + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D;

			/* 2-step output scheme */
			
			/* Count number of candidate edges */
			kernel_count_candidate_edges<<<warp_blocks_grid, BLOCK_SIZE1D>>>(ht.d_vertices_u, ht.d_addresses, ht.count_u, c_set, u, v,
                               q.d_vertices, q.d_labels, q.d_edges,
                               g.d_vertices, g.d_labels, g.d_edges);
				
			cudaDeviceSynchronize();

			thrust::device_ptr<int> dev_ptr(ht.d_addresses);
			
			/* Computing address of the first v′for u′*/
			thrust::inclusive_scan(dev_ptr, dev_ptr + ht.count_u + 1, dev_ptr); // in-place scan

			ht.count_v = dev_ptr[ht.count_u];
			
			cudaMalloc(&ht.d_vertices_v, ht.count_v * sizeof(int));

			kernel_fill_candidate_edges<<<warp_blocks_grid, BLOCK_SIZE1D>>>(ht.d_vertices_u, ht.d_addresses, ht.d_vertices_v, ht.count_u, c_set, u, v,
                               q.d_vertices, q.d_labels, q.d_edges,
                               g.d_vertices, g.d_labels, g.d_edges);	

			cudaDeviceSynchronize();
			E[std::make_pair(u, v)] = ht;
		}
        }

        return E;
}

void get_cmd_line_args(int argc, char **argv)
{
        enum class Args
        {
                DATA_GRAPH_FILE,
                QUERY_GRAPH_FILE,
                NONE
        };

        Args flag = Args::NONE;

        for (int i = 1; i < argc; i++) {
                std::string s(argv[i]);
                if (s.compare("--dataGraphFile") == 0)
                        flag = Args::DATA_GRAPH_FILE;
                else if (s.compare("--queryGraphFile") == 0)
                        flag = Args::QUERY_GRAPH_FILE;
                else {
                       	if (flag == Args::DATA_GRAPH_FILE)
                        	data_graph_file = s;
                        if (flag == Args::QUERY_GRAPH_FILE)
                                query_graph_file = s;
                        flag = Args::NONE;
                }
        }
}

int get_warp_size() {
	cudaDeviceProp device_prop;
	if (cudaSuccess != cudaGetDeviceProperties(&device_prop, 0)) {
                printf("Get device properties failed.\n");
                exit(EXIT_FAILURE);
        } else {
                return device_prop.warpSize;
        }
}

int main(int argc, char *argv[]) {

	/* Get command line arguments */
	get_cmd_line_args(argc, argv);

	Graph data_graph(9), query_graph(6);

	try {
		if (!data_graph_file.empty()) {	/* A hyphen or a filename is provided */
			data_graph.load(data_graph_file);
		}
		else {
			/* When no configuration is specified use Dijkstra’s original five-philosopher cycle */
			data_graph.reset(9);
			data_graph.add_label('B');
			data_graph.add_label('A');
			data_graph.add_label('B');
			data_graph.add_label('B');
			data_graph.add_label('A');
			data_graph.add_label('A');
			data_graph.add_label('C');
			data_graph.add_label('C');
			data_graph.add_label('A');
			
			data_graph.add_edge(0, 1);
			data_graph.add_edge(1, 2);
			data_graph.add_edge(1, 5);
			data_graph.add_edge(1, 6);
			data_graph.add_edge(2, 5);
			data_graph.add_edge(5, 6);
			data_graph.add_edge(2, 6);
			data_graph.add_edge(3, 6);
			data_graph.add_edge(6, 7);
			data_graph.add_edge(3, 7);
			data_graph.add_edge(3, 4);
			data_graph.add_edge(4, 7);
			data_graph.add_edge(4, 8);
		}

		if (!query_graph_file.empty()) { /* A hyphen or a filename is provided */
                        query_graph.load(query_graph_file);
                }
                else {
                        /* When no configuration is specified use Dijkstra’s original five-philosopher cycle */
                        query_graph.reset(6);
                        query_graph.add_label('B');
                        query_graph.add_label('A');
                        query_graph.add_label('B');
                        query_graph.add_label('A');
                        query_graph.add_label('C');
                        query_graph.add_label('C');
                     
                        query_graph.add_edge(0, 1);
                        query_graph.add_edge(1, 2);
                        query_graph.add_edge(1, 3);
                        query_graph.add_edge(1, 4);
                        query_graph.add_edge(2, 3);
                        query_graph.add_edge(2, 4);
                        query_graph.add_edge(3, 4);
                        query_graph.add_edge(4, 5);
                }

	}
	catch (const std::string& msg) {
		std::cerr << msg << std::endl;
		exit(EXIT_FAILURE);	
	}

	/* Check connectivity of the graph to proceed */
	if (!data_graph.is_connected() || !query_graph.is_connected()) {
		std::cerr << "Sorry one of our philosophers has no bottle to drink from.! \nExiting..." << std::endl;
		exit(EXIT_FAILURE);
	}

//	data_graph.print();
//	query_graph.print();

	data_graph.load_device();
	query_graph.load_device();

	Graph query_spanning_tree;

	init_spanning_tree(query_spanning_tree, query_graph);

	bool *c_set = initialize_candidate_vertices(query_spanning_tree, data_graph);
/*
	c_set = refine_candidate_vertices(c_set, query_graph, data_graph);
		
	std::map<std::pair<int, int>, hash_table> E = find_candidate_edges(c_set, query_graph, data_graph);

	join_candidate_edges();
*/
	cudaDeviceReset();


	return EXIT_SUCCESS;
}
