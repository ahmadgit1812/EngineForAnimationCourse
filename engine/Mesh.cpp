#include "Mesh.h"
#include <utility>
#include "ObjLoader.h"
#include "igl/edge_flaps.h"
#include "igl/parallel_for.h"
#include "igl/shortest_edge_and_midpoint.h"
#include "igl/collapse_edge.h"
#include <igl/vertex_triangle_adjacency.h>
#include "igl/per_vertex_normals.h"
#include <igl/per_face_normals.h>
#include <igl/decimate_callback_types.h>
#include <circulation.h>


namespace cg3d
{

Mesh::Mesh(std::string name, Eigen::MatrixXd vertices, Eigen::MatrixXi faces, Eigen::MatrixXd vertexNormals, Eigen::MatrixXd textureCoords)
        : name(std::move(name)), data{{vertices, faces, vertexNormals, textureCoords}} {
    Mesh::initMeshDataStructure();
}

void Mesh::initMeshDataStructure() {

        meshDataStructure.F = Eigen::MatrixXi(data[0].faces);
        meshDataStructure.V = Eigen::MatrixXd(data[0].vertices);
        igl::edge_flaps(meshDataStructure.F,meshDataStructure.E,meshDataStructure.EMAP,meshDataStructure.EF,meshDataStructure.EI);
        Q_Matrix_for_vertices_init();
        Q_heap_init();
        marked_edges = Eigen::VectorXi::Zero(meshDataStructure.E.rows());
        meshDataStructure.num_of_edges = meshDataStructure.E.rows();

        

}

void Mesh::tutorial_initMeshDataStructure(){
    meshDataStructure.F = Eigen::MatrixXi(data[0].faces);
    meshDataStructure.V = Eigen::MatrixXd(data[0].vertices);

    igl::edge_flaps(meshDataStructure.F,meshDataStructure.E,meshDataStructure.EMAP,meshDataStructure.EF,meshDataStructure.EI);
    meshDataStructure.C.resize(meshDataStructure.E.rows(),meshDataStructure.V.cols());
    Eigen::VectorXd costs(meshDataStructure.E.rows());
    // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
    // Q.clear();
    meshDataStructure.Q_tutorial = {};
    meshDataStructure.EQ_tutorial = Eigen::VectorXi::Zero(meshDataStructure.E.rows());
    {
        Eigen::VectorXd costs(meshDataStructure.E.rows());
        igl::parallel_for(meshDataStructure.E.rows(),[&](const int e)
        {
            double cost = e;
            Eigen::RowVectorXd p(1,3);
            igl::shortest_edge_and_midpoint(e,meshDataStructure.V,meshDataStructure.F,meshDataStructure.E,meshDataStructure.EMAP,meshDataStructure.EF,meshDataStructure.EI,cost,p);
            meshDataStructure.C.row(e) = p;
            costs(e) = cost;
        },10000);
        for(int e = 0;e<meshDataStructure.E.rows();e++)
        {
            meshDataStructure.Q_tutorial.emplace(costs(e),e,0);
        }
    }
}


void Mesh::C_Matrix_for_edges_init() {
    meshDataStructure.C.resize(meshDataStructure.E.rows(), meshDataStructure.V.cols());
    for(int i=0; i<meshDataStructure.E.rows(); i++){
        meshDataStructure.C.row(i) = 0.5*(meshDataStructure.V.row(meshDataStructure.E(i,0))+meshDataStructure.V.row(meshDataStructure.E(i,1)));
    }

}

void Mesh::Q_heap_init(){
    C_Matrix_for_edges_init();
    for(int i=0; i< meshDataStructure.E.rows(); i++){
        meshDataStructure.Q.push(std::tuple<double , int>(calculate_edge_cost(i), i));
    }
}

double Mesh::calculate_edge_cost(int edge_index){
    int edge_v0 = meshDataStructure.E(edge_index,0);
    int edge_v1 = meshDataStructure.E(edge_index,1);

    Eigen::Matrix4d edge_matrix = meshDataStructure.Q_Matrix_for_vertices[edge_v0]
            + meshDataStructure.Q_Matrix_for_vertices[edge_v1];

    Eigen::Matrix4d temp_matrix = Eigen::Matrix4d(edge_matrix);
    temp_matrix(3,0)=0;
    temp_matrix(3,1)=0;
    temp_matrix(3,2)=0;
    temp_matrix(3,3)=1;
    if(temp_matrix.determinant() != 0){
        Eigen::Vector4d new_vertex_position = temp_matrix.inverse() *Eigen::Vector4d(0,0,0,1);
        meshDataStructure.C(edge_index,0) =new_vertex_position(0);
        meshDataStructure.C(edge_index,1) =new_vertex_position(1);
        meshDataStructure.C(edge_index,2) =new_vertex_position(2);
        Eigen::RowVector4d new_vertex_position_transpose = new_vertex_position.transpose();
        return new_vertex_position_transpose * edge_matrix * new_vertex_position;
    }else{

        // middle vertex cost calc
        Eigen::Vector4d new_vertex_position =
                Eigen::Vector4d(meshDataStructure.C.row(edge_index)(0), meshDataStructure.C.row(edge_index)(1),
                                meshDataStructure.C.row(edge_index)(2), 1);

        Eigen::RowVector4d new_vertex_position_transpose = new_vertex_position.transpose();
        double middle_val = new_vertex_position_transpose * edge_matrix * new_vertex_position;

        // v0 vertex cost calc
        Eigen::Vector4d v0 = Eigen::Vector4d(meshDataStructure.V.row(edge_v0)(0), meshDataStructure.V.row(edge_v0)(1),
                                             meshDataStructure.V.row(edge_v0)(2), 1);
        Eigen::RowVector4d v0_transpose = v0.transpose();
        double v0_val = v0_transpose * meshDataStructure.Q_Matrix_for_vertices[edge_v0] * v0;

        // v1 vertex cost calc
        Eigen::Vector4d v1 = Eigen::Vector4d(meshDataStructure.V.row(edge_v1)(0), meshDataStructure.V.row(edge_v1)(1),
                                             meshDataStructure.V.row(edge_v1)(2), 1);

        Eigen::RowVector4d v1_transpose = v1.transpose();
        double v1_val = v1_transpose * meshDataStructure.Q_Matrix_for_vertices[edge_v1] * v1;

        // choosing lowest vertex cost
        int min_val = std::min(std::min(v0_val, v1_val), middle_val);
        if (min_val == v0_val) {
            meshDataStructure.C.row(edge_index) = meshDataStructure.V.row(edge_v0);
            return v0_val;
        }

        if (min_val == v1_val) {
            meshDataStructure.C.row(edge_index) = meshDataStructure.V.row(edge_v1);
            return v1_val;
        }
        return middle_val;
    }
}

void Mesh::Q_Matrix_for_vertices_init(){

    std::vector<std::vector<int>> vertex_adjacency;
    vertex_adjacency.resize(meshDataStructure.V.rows());
    for(int fi=0; fi<meshDataStructure.F.rows(); ++fi)
    {
        for(int i = 0; i < meshDataStructure.F.cols(); ++i)
        {
            vertex_adjacency[meshDataStructure.F(fi,i)].push_back(fi);

        }
    }
    Eigen::MatrixXd faces_normals;
    igl::per_face_normals(meshDataStructure.V, meshDataStructure.F, faces_normals);


    for(int i=0; i< meshDataStructure.V.rows(); i++){
        meshDataStructure.Q_Matrix_for_vertices.push_back(Eigen::Matrix4d::Zero());
        calculate_Q_matrix_for_vertex(i, vertex_adjacency[i], faces_normals);
    }

}

void Mesh::calculate_Q_matrix_for_vertex(int vertex_index, std::vector<int>& vertex_adjacency, Eigen::MatrixXd & faces_normals){

        for(int i=0; i<vertex_adjacency.size(); i++){
            Eigen::Vector3d curr_normal = faces_normals.row(vertex_adjacency[i]).normalized();

            Eigen::MatrixXd p;
            p.resize(4,1);

            p << curr_normal[0] , curr_normal[1], curr_normal[2],((meshDataStructure.V.row(vertex_index) * curr_normal)*-1);

            Eigen::MatrixXd p_transpose = p.transpose();
            Eigen::Matrix4d kp = p * p_transpose;
            meshDataStructure.Q_Matrix_for_vertices[vertex_index] = meshDataStructure.Q_Matrix_for_vertices[vertex_index] + kp;
        }
}

int Mesh::tutorial_simplify(int num_of_faces){

        int num_collapsed=0;
        bool something_collapsed = false;
        int data_max_index = -1;
        if(!meshDataStructure.Q_tutorial.empty())
        {
            // collapse edge
            const int max_iter = num_of_faces;
            for(int j = 0;j<max_iter;j++)
            {
                if(!igl::collapse_edge(igl::shortest_edge_and_midpoint,meshDataStructure.V,meshDataStructure.F,meshDataStructure.E,meshDataStructure.EMAP,meshDataStructure.EF,meshDataStructure.EI,meshDataStructure.Q_tutorial,meshDataStructure.EQ_tutorial,meshDataStructure.C))
                    break;

                something_collapsed = true;
                num_collapsed++;
            }

            if(something_collapsed)
             {

                 Eigen::MatrixXd vertexNormalsTemp;
                 vertexNormalsTemp.resize(0,0);
                 Eigen::MatrixXd textureCoordsTemp;
                 textureCoordsTemp.resize(0,0);
                 textureCoordsTemp = data[0].textureCoords;

                 igl::per_vertex_normals(meshDataStructure.V, meshDataStructure.F, vertexNormalsTemp);
                 data.push_back({Eigen::MatrixXd(meshDataStructure.V), Eigen::MatrixXi(meshDataStructure.F), vertexNormalsTemp , textureCoordsTemp});
                 data_max_index = data.size()-1;
             }

        }
        return data_max_index;
}


int Mesh::simplify(int num_of_faces){
    int num_collapsed=0;
    bool something_collapsed = false;
    int data_max_index = -1;

    if(!meshDataStructure.Q.empty())
    {
        for(int j = 0;j<num_of_faces;j++)
        {
            if(!collapse_edge())
                break;


            something_collapsed = true;
            num_collapsed++;
        }
        if(something_collapsed)
        {

            Eigen::MatrixXd vertexNormalsTemp;
            vertexNormalsTemp.resize(0,0);
            Eigen::MatrixXd textureCoordsTemp;
            textureCoordsTemp.resize(0,0);
            textureCoordsTemp = data[0].textureCoords;

            igl::per_vertex_normals(meshDataStructure.V, meshDataStructure.F, vertexNormalsTemp);
            data.push_back({Eigen::MatrixXd(meshDataStructure.V), Eigen::MatrixXi(meshDataStructure.F), vertexNormalsTemp , textureCoordsTemp});
            data_max_index = data.size()-1;
        }
    }
    return data_max_index;
}

bool Mesh::collapse_edge(){

    if(meshDataStructure.Q.empty()) return false;
    std::tuple<double , int> min_edge = meshDataStructure.Q.top();
    meshDataStructure.Q.pop();


    double min_edge_cost = std::get<0>(min_edge);
    int min_edge_index = std::get<1>(min_edge);


    if(min_edge_cost==std::numeric_limits<double>::infinity())
        return false;

    if(marked_edges[min_edge_index] == -1)
        return collapse_edge();

    if(marked_edges[min_edge_index] == 1)
    {
        update_marked_edge(min_edge_index);
        return collapse_edge();
    }

    int first_vertex_index = meshDataStructure.E(min_edge_index, 0);
    int sec_vertex_index = meshDataStructure.E(min_edge_index, 1);

    std::vector<int> surrounding_faces1 = igl::circulation(min_edge_index, true, meshDataStructure.EMAP, meshDataStructure.EF, meshDataStructure.EI);

    std::vector<int> surrounding_faces2 = igl::circulation(min_edge_index, false, meshDataStructure.EMAP, meshDataStructure.EF, meshDataStructure.EI);

    int f1,f2 ,e1,e2;

    Eigen::Vector3d new_v_position = meshDataStructure.C.row(min_edge_index);
    if(igl::collapse_edge(min_edge_index, new_v_position, meshDataStructure.V,
                          meshDataStructure.F, meshDataStructure.E, meshDataStructure.EMAP,
                          meshDataStructure.EF, meshDataStructure.EI, e1, e2, f1, f2)){

        marked_edges[min_edge_index] = -1;
        marked_edges[e1] = -1;
        marked_edges[e2] = -1;
        meshDataStructure.num_of_edges = meshDataStructure.num_of_edges-3;

        Eigen::Matrix4d Q12 = meshDataStructure.Q_Matrix_for_vertices[first_vertex_index] + meshDataStructure.Q_Matrix_for_vertices[sec_vertex_index];
        meshDataStructure.Q_Matrix_for_vertices[first_vertex_index] = Q12;
        meshDataStructure.Q_Matrix_for_vertices[sec_vertex_index] = Q12;

        for(int i=0; i <surrounding_faces1.size(); i++)
                mark_face_edges(surrounding_faces1[i],first_vertex_index,sec_vertex_index);

        for(int i=1; i <surrounding_faces2.size()-1; i++)
                mark_face_edges(surrounding_faces2[i],first_vertex_index,sec_vertex_index);

        std::cout <<"edge "<< min_edge_index<< ", cost = "<< min_edge_cost << ", new v position ";
        std::cout << "("<< new_v_position(0)<<","<<new_v_position(1)<<","<<new_v_position(2)<< ")"<<std::endl;
        return true;

    }
    else{
        meshDataStructure.Q.push(std::tuple<double , int>(std::numeric_limits<double>::infinity(), min_edge_index) );
        return false;
    }

}

void Mesh::update_marked_edge(int edge_index){
    meshDataStructure.C.row(edge_index) = 0.5*(meshDataStructure.V.row(meshDataStructure.E(edge_index,0))+meshDataStructure.V.row(meshDataStructure.E(edge_index,1)));
    marked_edges[edge_index]=0;
    meshDataStructure.Q.push(std::tuple<double , int>(calculate_edge_cost(edge_index), edge_index));
}

void Mesh::mark_face_edges(int face_index, int first_vertex_index, int sec_vertex_index){
    if(!((meshDataStructure.F(face_index,0) ==meshDataStructure.F(face_index,1)) &&
         (meshDataStructure.F(face_index,1) ==meshDataStructure.F(face_index,2))
         && (meshDataStructure.F(face_index,2) == 0 ))) {

        for (int i = 0; i < 3; i++) {
            int num_of_faces = meshDataStructure.F.rows();
            int edge_index = meshDataStructure.EMAP(face_index + (i * num_of_faces));
            bool edge_is_connected_to_v1 = meshDataStructure.E(edge_index, 0) == first_vertex_index ||
                                           meshDataStructure.E(edge_index, 1) == first_vertex_index;
            bool edge_is_connected_to_v2 = meshDataStructure.E(edge_index, 0) == sec_vertex_index ||
                                           meshDataStructure.E(edge_index, 1) == sec_vertex_index;
            if (marked_edges[edge_index] != -1 && (edge_is_connected_to_v1 || edge_is_connected_to_v2))
                marked_edges[edge_index] = 1;
        }
    }

}





const std::shared_ptr<Mesh>& Mesh::Plane()
{
    static auto data = std::istringstream(R"(
v 1.000000 1.000000 0.000000
v -1.000000 1.000000 0.000000
v 1.000000 -1.000000 0.000000
v -1.000000 -1.000000 0.000000
vt 1.000000 0.000000
vt 0.000000 1.000000
vt 0.000000 0.000000
vt 1.000000 1.000000
vn 1.0000 -0.0000 0.0000
s off
f 2/1/1 3/2/1 1/3/1
f 2/1/1 4/4/1 3/2/1
        )");

    static const auto MESH = ObjLoader::MeshFromObj("Plane", data);

    return MESH;
}

const std::shared_ptr<Mesh>& Mesh::Cube()
{
    static auto data = std::istringstream(R"(
# This file uses centimeters as units for non-parametric coordinates.
v -0.500000 -0.500000 0.500000
v 0.500000 -0.500000 0.500000
v -0.500000 0.500000 0.500000
v 0.500000 0.500000 0.500000
v -0.500000 0.500000 -0.500000
v 0.500000 0.500000 -0.500000
v -0.500000 -0.500000 -0.500000
v 0.500000 -0.500000 -0.500000
vt 0.0 0.0
vt 0.0 1.0
vt 1.0 0.0
vt 1.0 1.0
f 1/1 2/2 3/3
f 3/3 2/2 4/4
f 3/1 4/2 5/3
f 5/3 4/2 6/4
f 5/1 6/2 7/3
f 7/3 6/2 8/4
f 7/1 8/2 1/3
f 1/3 8/2 2/4
f 2/1 8/2 4/3
f 4/3 8/2 6/4
f 7/1 1/2 5/3
f 5/3 1/2 3/4
        )");

    static const auto MESH = ObjLoader::MeshFromObj("Cube", data);

    return MESH;
}

const std::shared_ptr<Mesh>& Mesh::Octahedron()
{
    static auto data = std::istringstream(R"(
v 0.000000 1.000000 0.000000
v 1.000000 0.000000 0.000000
v 0.000000 0.000000 -1.000000
v -1.000000 0.000000 0.000000
v 0.000000 -0.000000 1.000000
v 0.000000 -1.000000 -0.000000
s off
f 1 2 3
f 1 3 4
f 1 4 5
f 1 5 2
f 6 3 2
f 6 4 3
f 6 5 4
f 6 2 5
        )");

    static const auto MESH = ObjLoader::MeshFromObj("Octahedron", data);

    return MESH;
}

const std::shared_ptr<Mesh>& Mesh::Tetrahedron()
{
    static auto data = std::istringstream(R"(
v 1 0 0
v 0 1 0
v 0 0 1
v 0 0 0
f 2 4 3
f 4 2 1
f 3 1 2
f 1 3 4
        )");

    static const auto MESH = ObjLoader::MeshFromObj("Tetrahedron", data);

    return MESH;
}

const std::shared_ptr<Mesh>& Mesh::Cylinder()
{
    static auto data = std::istringstream(R"(
# This file uses centimeters as units for non-parametric coordinates.

v -0.800000 -0.403499 -0.131105
v -0.800000 -0.343237 -0.249376
v -0.800000 -0.249376 -0.343237
v -0.800000 -0.131105 -0.403499
v -0.800000 0.000000 -0.424264
v -0.800000 0.131105 -0.403499
v -0.800000 0.249376 -0.343237
v -0.800000 0.343237 -0.249376
v -0.800000 0.403499 -0.131105
v -0.800000 0.424264 0.000000
v -0.800000 0.403499 0.131105
v -0.800000 0.343237 0.249376
v -0.800000 0.249376 0.343237
v -0.800000 0.131105 0.403499
v -0.800000 0.000000 0.424264
v -0.800000 -0.131105 0.403499
v -0.800000 -0.249376 0.343237
v -0.800000 -0.343237 0.249376
v -0.800000 -0.403499 0.131105
v -0.800000 -0.424264 0.000000
v 0.800000 -0.403499 -0.131105
v 0.800000 -0.343237 -0.249376
v 0.800000 -0.249376 -0.343237
v 0.800000 -0.131105 -0.403499
v 0.800000 -0.000000 -0.424264
v 0.800000 0.131105 -0.403499
v 0.800000 0.249376 -0.343237
v 0.800000 0.343237 -0.249376
v 0.800000 0.403499 -0.131105
v 0.800000 0.424264 0.000000
v 0.800000 0.403499 0.131105
v 0.800000 0.343237 0.249376
v 0.800000 0.249376 0.343237
v 0.800000 0.131105 0.403499
v 0.800000 0.000000 0.424264
v 0.800000 -0.131105 0.403499
v 0.800000 -0.249376 0.343237
v 0.800000 -0.343237 0.249376
v 0.800000 -0.403499 0.131105
v 0.800000 -0.424264 0.000000
v -0.800000 0.000000 0.000000
v 0.800000 -0.000000 0.000000
vt 0.648603 0.107966
vt 0.626409 0.064408
vt 0.591842 0.029841
vt 0.548284 0.007647
vt 0.500000 -0.000000
vt 0.451716 0.007647
vt 0.408159 0.029841
vt 0.373591 0.064409
vt 0.351397 0.107966
vt 0.343750 0.156250
vt 0.351397 0.204534
vt 0.373591 0.248091
vt 0.408159 0.282659
vt 0.451716 0.304853
vt 0.500000 0.312500
vt 0.548284 0.304853
vt 0.591841 0.282659
vt 0.626409 0.248091
vt 0.648603 0.204534
vt 0.656250 0.156250
vt 0.375000 0.312500
vt 0.387500 0.312500
vt 0.400000 0.312500
vt 0.412500 0.312500
vt 0.425000 0.312500
vt 0.437500 0.312500
vt 0.450000 0.312500
vt 0.462500 0.312500
vt 0.475000 0.312500
vt 0.487500 0.312500
vt 0.500000 0.312500
vt 0.512500 0.312500
vt 0.525000 0.312500
vt 0.537500 0.312500
vt 0.550000 0.312500
vt 0.562500 0.312500
vt 0.575000 0.312500
vt 0.587500 0.312500
vt 0.600000 0.312500
vt 0.612500 0.312500
vt 0.625000 0.312500
vt 0.375000 0.688440
vt 0.387500 0.688440
vt 0.400000 0.688440
vt 0.412500 0.688440
vt 0.425000 0.688440
vt 0.437500 0.688440
vt 0.450000 0.688440
vt 0.462500 0.688440
vt 0.475000 0.688440
vt 0.487500 0.688440
vt 0.500000 0.688440
vt 0.512500 0.688440
vt 0.525000 0.688440
vt 0.537500 0.688440
vt 0.550000 0.688440
vt 0.562500 0.688440
vt 0.575000 0.688440
vt 0.587500 0.688440
vt 0.600000 0.688440
vt 0.612500 0.688440
vt 0.625000 0.688440
vt 0.648603 0.795466
vt 0.626409 0.751908
vt 0.591842 0.717341
vt 0.548284 0.695147
vt 0.500000 0.687500
vt 0.451716 0.695147
vt 0.408159 0.717341
vt 0.373591 0.751909
vt 0.351397 0.795466
vt 0.343750 0.843750
vt 0.351397 0.892034
vt 0.373591 0.935591
vt 0.408159 0.970159
vt 0.451716 0.992353
vt 0.500000 1.000000
vt 0.548284 0.992353
vt 0.591841 0.970159
vt 0.626409 0.935591
vt 0.648603 0.892034
vt 0.656250 0.843750
vt 0.500000 0.150000
vt 0.500000 0.837500
f 1/21 2/22 21/42
f 21/42 2/22 22/43
f 2/22 3/23 22/43
f 22/43 3/23 23/44
f 3/23 4/24 23/44
f 23/44 4/24 24/45
f 4/24 5/25 24/45
f 24/45 5/25 25/46
f 5/25 6/26 25/46
f 25/46 6/26 26/47
f 6/26 7/27 26/47
f 26/47 7/27 27/48
f 7/27 8/28 27/48
f 27/48 8/28 28/49
f 8/28 9/29 28/49
f 28/49 9/29 29/50
f 9/29 10/30 29/50
f 29/50 10/30 30/51
f 10/30 11/31 30/51
f 30/51 11/31 31/52
f 11/31 12/32 31/52
f 31/52 12/32 32/53
f 12/32 13/33 32/53
f 32/53 13/33 33/54
f 13/33 14/34 33/54
f 33/54 14/34 34/55
f 14/34 15/35 34/55
f 34/55 15/35 35/56
f 15/35 16/36 35/56
f 35/56 16/36 36/57
f 16/36 17/37 36/57
f 36/57 17/37 37/58
f 17/37 18/38 37/58
f 37/58 18/38 38/59
f 18/38 19/39 38/59
f 38/59 19/39 39/60
f 19/39 20/40 39/60
f 39/60 20/40 40/61
f 20/40 1/41 40/61
f 40/61 1/41 21/62
f 2/2 1/1 41/83
f 3/3 2/2 41/83
f 4/4 3/3 41/83
f 5/5 4/4 41/83
f 6/6 5/5 41/83
f 7/7 6/6 41/83
f 8/8 7/7 41/83
f 9/9 8/8 41/83
f 10/10 9/9 41/83
f 11/11 10/10 41/83
f 12/12 11/11 41/83
f 13/13 12/12 41/83
f 14/14 13/13 41/83
f 15/15 14/14 41/83
f 16/16 15/15 41/83
f 17/17 16/16 41/83
f 18/18 17/17 41/83
f 19/19 18/18 41/83
f 20/20 19/19 41/83
f 1/1 20/20 41/83
f 21/81 22/80 42/84
f 22/80 23/79 42/84
f 23/79 24/78 42/84
f 24/78 25/77 42/84
f 25/77 26/76 42/84
f 26/76 27/75 42/84
f 27/75 28/74 42/84
f 28/74 29/73 42/84
f 29/73 30/72 42/84
f 30/72 31/71 42/84
f 31/71 32/70 42/84
f 32/70 33/69 42/84
f 33/69 34/68 42/84
f 34/68 35/67 42/84
f 35/67 36/66 42/84
f 36/66 37/65 42/84
f 37/65 38/64 42/84
f 38/64 39/63 42/84
f 39/63 40/82 42/84
f 40/82 21/81 42/84
        )");

    static const auto MESH = ObjLoader::MeshFromObj("Cylinder", data);

    return MESH;
}

} // namespace cg3d
