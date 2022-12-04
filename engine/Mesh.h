#pragma once

#include <Eigen/Core>
#include <iostream>
#include <utility>
#include <vector>
#include <igl/min_heap.h>


namespace cg3d
{


struct MeshDataStructure
{
    Eigen::MatrixXd V; // Vertices of the mesh (#V x 3)
    Eigen::MatrixXi F; // Faces of the mesh (#F x 3)
    Eigen::MatrixXi E; // edges <index of source vertex, index of des/na/on vertex>
    Eigen::VectorXi EMAP; // connects faces to edges
    Eigen::MatrixXi EF; // connects edges to faces
    Eigen::MatrixXi EI; // connects edge to vertex index in triangle (0,1,2)
    igl::min_heap< std::tuple<double,int> > Q; // priority queue Q
    std::vector<Eigen::Matrix4d > Q_Matrix_for_vertices;
    Eigen::MatrixXd C; // position of the new vertex after collapsing the corresponding edge (in model coordinates).
    int num_of_edges;


    // tutorial data
    igl::min_heap< std::tuple<double,int, int> > Q_tutorial; // priority queue Q
    Eigen::VectorXi EQ_tutorial; // maybe should delete
};

struct MeshData
{
    const Eigen::MatrixXd vertices; // Vertices of the mesh (#V x 3)
    const Eigen::MatrixXi faces; // Faces of the mesh (#F x 3)
    const Eigen::MatrixXd vertexNormals; // One normal per vertex
    const Eigen::MatrixXd textureCoords; // UV vertices
};

class Mesh
{
public:
    std::string name;
    std::vector<MeshData> data;
    MeshDataStructure meshDataStructure;

    Mesh(std::string name, Eigen::MatrixXd vertices, Eigen::MatrixXi faces, Eigen::MatrixXd vertexNormals, Eigen::MatrixXd textureCoords);
    Mesh(std::string name, std::vector<MeshData> data) : name(std::move(name)), data(std::move(data)) { initMeshDataStructure();};
    Mesh(const Mesh& mesh) = default;


    static const std::shared_ptr<Mesh>& Plane();
    static const std::shared_ptr<Mesh>& Cube();
    static const std::shared_ptr<Mesh>& Tetrahedron();
    static const std::shared_ptr<Mesh>& Octahedron();
    static const std::shared_ptr<Mesh>& Cylinder();


    int simplify(int num_of_faces);
    int tutorial_simplify(int num_of_faces);


private:

    void initMeshDataStructure();
    void tutorial_initMeshDataStructure();
    void Q_Matrix_for_vertices_init();
    void calculate_Q_matrix_for_vertex(int vertex_index, std::vector<int>& vertex_adjacency, Eigen::MatrixXd & faces_normals);

    void C_Matrix_for_edges_init();
    void Q_heap_init();
    double calculate_edge_cost(int edge_index);
    bool collapse_edge();

    void mark_face_edges(int face_index, int first_vertex_index, int sec_vertex_index);
    void update_marked_edge(int edge_index);


    Eigen::VectorXi marked_edges;
};

} // namespace cg3d
